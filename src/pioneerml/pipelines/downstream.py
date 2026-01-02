import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from torch_geometric.data import Data, Batch

from pioneerml.data.datasets.graph_group import GraphRecord
from pioneerml.data.datasets.utils import fully_connected_edge_index, build_edge_attr
from pioneerml.models.classifiers import GroupSplitter
from pioneerml.models.regressors import PionStopRegressor, PositronAngleModel
from pioneerml.data.event_mixer import EventContainer


class DownstreamPipeline:
    def __init__(self, device: torch.device):
        self.device = device

    def load_models(
        self,
        splitter_path: str,
        pi_stop_path: str,
        pos_angle_path: str,
    ):
        self.splitter = GroupSplitter().to(self.device)
        self.splitter.load_state_dict(torch.load(splitter_path, map_location=self.device))
        self.splitter.eval()

        self.pi_stop = PionStopRegressor().to(self.device)
        self.pi_stop.load_state_dict(torch.load(pi_stop_path, map_location=self.device))
        self.pi_stop.eval()

        self.pos_angle = PositronAngleModel().to(self.device)
        self.pos_angle.load_state_dict(torch.load(pos_angle_path, map_location=self.device))
        self.pos_angle.eval()

    def _prepare_standard_input(self, record: GraphRecord, subset_indices: Optional[np.ndarray] = None) -> Data:
        """
        Prepares standard 4-channel input [coord, z, energy, view].
        Optionally subsets by indices (e.g. for Pion component).
        """
        if subset_indices is not None:
            coord = torch.tensor(record.coord[subset_indices], dtype=torch.float, device=self.device)
            z = torch.tensor(record.z[subset_indices], dtype=torch.float, device=self.device)
            energy = torch.tensor(record.energy[subset_indices], dtype=torch.float, device=self.device)
            view = torch.tensor(record.view[subset_indices], dtype=torch.float, device=self.device)
        else:
            coord = torch.tensor(record.coord, dtype=torch.float, device=self.device)
            z = torch.tensor(record.z, dtype=torch.float, device=self.device)
            energy = torch.tensor(record.energy, dtype=torch.float, device=self.device)
            view = torch.tensor(record.view, dtype=torch.float, device=self.device)

        node_features = torch.stack([coord, z, energy, view], dim=1)
        edge_index = fully_connected_edge_index(len(coord), device=self.device)
        edge_attr = build_edge_attr(node_features, edge_index)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        data.u = torch.tensor([[energy.sum()]], dtype=torch.float, device=self.device)
        data.batch = torch.zeros(len(coord), dtype=torch.long, device=self.device)
        return data

    def process_event(self, event: EventContainer) -> EventContainer:
        """
        Runs downstream tasks: Splitter, PionStop, Matching, Angle.
        Uses Batch processing for efficiency.
        """
        records = event.records
        if not records:
            return event

        splitter_indices = []
        pion_indices = []
        positron_indices = []

        for i, record in enumerate(records):
            if not hasattr(record, "group_probs") or record.group_probs is None:
                continue
            probs = record.group_probs
            is_pion = probs[0] > 0.5
            is_muon = probs[1] > 0.5
            is_mip = probs[2] > 0.5

            if is_pion and is_muon and not is_mip:
                splitter_indices.append(i)
            elif is_pion:
                pion_indices.append(i)
            elif is_mip:
                positron_indices.append(i)

        if splitter_indices:
            split_data_list = [self._prepare_standard_input(records[i]) for i in splitter_indices]
            split_batch = Batch.from_data_list(split_data_list).to(self.device)

            with torch.no_grad():
                node_logits, energy_preds = self.splitter(split_batch)

                ptr = split_batch.ptr.cpu().numpy()
                energy_preds = energy_preds.cpu().numpy()
                node_logits = node_logits.cpu().numpy()

                for idx, split_idx in enumerate(splitter_indices):
                    rec = records[split_idx]
                    start, end = ptr[idx], ptr[idx + 1]

                    logits_slice = node_logits[start:end]
                    hit_probs = F.softmax(torch.tensor(logits_slice), dim=1).numpy()
                    hit_preds = hit_probs.argmax(axis=1)

                    rec.hit_pdgs = hit_preds
                    rec.split_energies = energy_preds[idx].tolist()

                    pion_mask = hit_preds == 0
                    muon_mask = hit_preds == 1

                    rec.split_pion_mask = pion_mask

                    if pion_mask.any():
                        current_pion_E = rec.energy[pion_mask].sum()
                        pred_pion_E = energy_preds[idx, 0]
                        if current_pion_E > 0:
                            rec.energy[pion_mask] *= pred_pion_E / current_pion_E

                    if muon_mask.any():
                        current_muon_E = rec.energy[muon_mask].sum()
                        pred_muon_E = energy_preds[idx, 1]
                        if current_muon_E > 0:
                            rec.energy[muon_mask] *= pred_muon_E / current_muon_E

                    if pion_mask.any():
                        pion_indices.append(split_idx)

        if pion_indices:
            pi_data_list = []
            valid_pi_indices = []

            for i in pion_indices:
                rec = records[i]
                subset = None
                if hasattr(rec, "split_pion_mask"):
                    subset = np.where(rec.split_pion_mask)[0]
                    if len(subset) == 0:
                        continue

                data = self._prepare_standard_input(rec, subset_indices=subset)
                pi_data_list.append(data)
                valid_pi_indices.append(i)

            if pi_data_list:
                pi_batch = Batch.from_data_list(pi_data_list).to(self.device)
                with torch.no_grad():
                    pi_stops = self.pi_stop(pi_batch).cpu().numpy()

                for k, rec_idx in enumerate(valid_pi_indices):
                    records[rec_idx].pred_pion_stop = pi_stops[k]

        if positron_indices:
            pos_data_list = []
            valid_pos_indices = []

            for i in positron_indices:
                rec = records[i]

                if hasattr(rec, "matched_pion_index") and rec.matched_pion_index is not None:
                    pi_idx = rec.matched_pion_index
                    if 0 <= pi_idx < len(records) and hasattr(records[pi_idx], "pred_pion_stop"):
                        pi_stop = records[pi_idx].pred_pion_stop
                        rec.pion_stop_for_angle = pi_stop

                        data = self._prepare_standard_input(rec)
                        data.pion_stop = torch.tensor(pi_stop, dtype=torch.float, device=self.device).unsqueeze(0)
                        pos_data_list.append(data)
                        valid_pos_indices.append(i)
                elif hasattr(rec, "pion_stop_for_angle") and rec.pion_stop_for_angle is not None:
                    data = self._prepare_standard_input(rec)
                    data.pion_stop = torch.tensor(rec.pion_stop_for_angle, dtype=torch.float, device=self.device).unsqueeze(0)
                    pos_data_list.append(data)
                    valid_pos_indices.append(i)

            if pos_data_list:
                pos_batch = Batch.from_data_list(pos_data_list).to(self.device)
                with torch.no_grad():
                    angles = self.pos_angle(pos_batch).cpu().numpy()

                for k, rec_idx in enumerate(valid_pos_indices):
                    records[rec_idx].pred_angle = angles[k]
                    records[rec_idx].total_energy = records[rec_idx].energy.sum()

        return event
