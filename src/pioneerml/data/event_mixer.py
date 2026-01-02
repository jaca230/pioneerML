"""
Utilities for mixing events and building mixed datasets for the EventBuilder.
Ported from deprecated/omar_pioneerML with minor adjustments for arc-length defaults.
"""

from __future__ import annotations

import numpy as np
import torch
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

from pioneerml.data.datasets.graph_group import GraphRecord


class EventContainer:
    """
    A container for a mixed event, holding a list of GraphRecords (one for each time group)
    and their corresponding origin IDs (0=main, 1=pileup1, 2=pileup2).
    """

    def __init__(self, records: List[GraphRecord], origins: List[int]):
        self.records = records
        self.origins = origins

    def __iter__(self):
        return iter(zip(self.records, self.origins))

    def __len__(self):
        return len(self.records)


class EventMixer:
    def __init__(self, hits_files: List[str], info_files: List[str], max_events: Optional[int] = None):
        """
        Initialize with lists of paths to hits and info NPY files.
        Files should correspond 1-to-1.
        :param max_events: Maximum number of events to load.
        """
        self.events = self.load_and_group_data(hits_files, info_files, max_events)

    def load_and_group_data(
        self, hits_files: List[str], info_files: List[str], max_events: Optional[int] = None
    ) -> List[List[GraphRecord]]:
        """
        Loads batches and reconstructs events from flattened groups.
        Returns a list of events. Each event is a list of GraphRecords.
        """
        all_events_map = defaultdict(list)

        hits_files = sorted(hits_files)
        info_files = sorted(info_files)

        if len(hits_files) != len(info_files):
            raise ValueError(f"Mismatch in number of files: {len(hits_files)} hits files vs {len(info_files)} info files")

        print(f"Loading {len(hits_files)} file pairs...")

        for h_file, i_file in zip(hits_files, info_files):
            if max_events is not None and len(all_events_map) >= max_events:
                print(f"Reached max_events limit ({max_events}). Stopping load.")
                break

            print(f"Loading {os.path.basename(h_file)} and {os.path.basename(i_file)}")
            hits_batch = np.load(h_file, allow_pickle=True)
            info_batch = np.load(i_file, allow_pickle=True)

            if len(hits_batch) != len(info_batch):
                print(f"Warning: Batch length mismatch in {h_file}. Hits: {len(hits_batch)}, Info: {len(info_batch)}")
                continue

            for hits, info in zip(hits_batch, info_batch):
                if len(info) < 12:
                    continue

                event_id = int(info[11])
                record = self._create_graph_record_from_raw(hits, info)
                all_events_map[event_id].append(record)

        events = list(all_events_map.values())
        if max_events is not None:
            events = events[:max_events]

        print(f"Reconstructed {len(events)} events.")
        return events

    def _create_graph_record_from_raw(self, hits: Any, info: np.ndarray) -> GraphRecord:
        """
        Converts raw hits and info into a GraphRecord.
        """
        hits = np.array(hits, dtype=float)

        if hits.ndim == 1:
            return GraphRecord(
                coord=np.array([]), z=np.array([]), energy=np.array([]), view=np.array([]),
                hit_pdgs=np.array([]), event_id=-1, group_id=-1
            )

        coord = hits[:, 0]
        z = hits[:, 1]
        view = hits[:, 2].astype(float)
        energy = hits[:, 3].astype(float)
        hit_pdgs = hits[:, 4].astype(int)

        pion_stop = info[3:6]
        theta = info[9]
        phi = info[10]
        event_id = int(info[11])

        if theta != -1000 and phi != -1000:
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            sin_p = np.sin(phi)
            cos_p = np.cos(phi)
            angle_vector = [sin_t * cos_p, sin_t * sin_p, cos_t]
        else:
            angle_vector = None

        return GraphRecord(
            coord=coord,
            z=z,
            energy=energy,
            view=view,
            hit_pdgs=hit_pdgs,
            event_id=0,  # Placeholder, will be set during mixing
            group_id=event_id,
            true_pion_stop=pion_stop if np.any(pion_stop != 0) else None,
            true_angle_vector=angle_vector,
        )

    def mix_event(self, main_event_idx: int) -> EventContainer:
        """
        Mixes the event at main_event_idx with random pileup events.
        """
        if main_event_idx >= len(self.events):
            raise IndexError("Event index out of bounds")

        time_groups = self.events[main_event_idx]
        mixed_groups: List[Tuple[GraphRecord, int]] = []

        # 1. Main pion groups (origin 0) -> all except last
        if len(time_groups) > 1:
            for g in time_groups[:-1]:
                mixed_groups.append((g, 0))

        inc_pos = False

        # 2. 20% chance positron from main event (origin 0)
        if np.random.rand() < 0.2 and time_groups:
            inc_pos = True
            mixed_groups.append((time_groups[-1], 0))

        # 3. Random event 1
        rand_idx1 = np.random.randint(len(self.events))
        random_event_1 = self.events[rand_idx1]

        if np.random.rand() < 0.2:
            if len(random_event_1) > 1:
                for g in random_event_1[:-1]:
                    mixed_groups.append((g, 1))

            if np.random.rand() < 0.2 and random_event_1:
                inc_pos = True
                mixed_groups.append((random_event_1[-1], 1))

        # 4. Random event 2
        rand_idx2 = np.random.randint(len(self.events))
        random_event_2 = self.events[rand_idx2]

        if np.random.rand() < 0.1 and random_event_2:
            inc_pos = True
            mixed_groups.append((random_event_2[-1], 2))

        # 5. If no positron included yet, force include main event positron
        if not inc_pos and time_groups:
            mixed_groups.append((time_groups[-1], 0))

        np.random.shuffle(mixed_groups)

        final_records: List[GraphRecord] = []
        final_origins: List[int] = []

        for record, origin in mixed_groups:
            final_records.append(record)
            final_origins.append(origin)

        return EventContainer(final_records, final_origins)

    def generate_mixed_batch(self, batch_size: int = 32) -> List[EventContainer]:
        """
        Generates a batch of mixed events.
        """
        mixed_batch = []
        for _ in range(batch_size):
            idx = np.random.randint(len(self.events))
            mixed_batch.append(self.mix_event(idx))
        return mixed_batch


def save_mixed_events(events: List[EventContainer], path: str):
    """
    Saves a list of EventContainer objects to a file using efficient columnar storage.
    Flattens the object hierarchy into concatenated tensors.
    """
    if not events:
        print("No events to save.")
        return

    all_coords = []
    all_zs = []
    all_energies = []
    all_views = []
    all_hit_pdgs = []

    all_record_event_ids = []
    all_record_group_ids = []
    all_record_true_pion_stops = []
    all_record_true_angle_vectors = []

    all_record_group_probs = []
    all_record_pred_endpoints = []

    all_record_hit_counts = []

    event_record_counts = []
    all_event_origins = []

    print("Flattening data...")
    for event in events:
        event_record_counts.append(len(event.records))

        for record, origin in zip(event.records, event.origins):
            coords = record.coord if isinstance(record.coord, np.ndarray) else np.array(record.coord)
            zs = record.z if isinstance(record.z, np.ndarray) else np.array(record.z)
            energies = record.energy if isinstance(record.energy, np.ndarray) else np.array(record.energy)
            views = record.view if isinstance(record.view, np.ndarray) else np.array(record.view)
            pdgs = record.hit_pdgs if isinstance(record.hit_pdgs, np.ndarray) else np.array(record.hit_pdgs)

            num_hits = len(coords)
            all_record_hit_counts.append(num_hits)

            all_coords.append(torch.tensor(coords, dtype=torch.float32))
            all_zs.append(torch.tensor(zs, dtype=torch.float32))
            all_energies.append(torch.tensor(energies, dtype=torch.float32))
            all_views.append(torch.tensor(views, dtype=torch.float32))
            all_hit_pdgs.append(torch.tensor(pdgs, dtype=torch.long))

            all_record_event_ids.append(record.event_id if record.event_id is not None else -1)
            all_record_group_ids.append(record.group_id if record.group_id is not None else -1)

            if record.true_pion_stop is not None:
                all_record_true_pion_stops.append(torch.tensor(record.true_pion_stop, dtype=torch.float32))
            else:
                all_record_true_pion_stops.append(None)

            if record.true_angle_vector is not None:
                all_record_true_angle_vectors.append(torch.tensor(record.true_angle_vector, dtype=torch.float32))
            else:
                all_record_true_angle_vectors.append(None)

            if record.group_probs is not None:
                all_record_group_probs.append(torch.tensor(record.group_probs, dtype=torch.float32))
            else:
                all_record_group_probs.append(None)

            if record.pred_endpoints is not None:
                all_record_pred_endpoints.append(torch.tensor(record.pred_endpoints, dtype=torch.float32))
            else:
                all_record_pred_endpoints.append(None)

            all_event_origins.append(origin)

    print("Concatenating tensors...")
    if all_coords:
        big_coords = torch.cat(all_coords)
        big_zs = torch.cat(all_zs)
        big_energies = torch.cat(all_energies)
        big_views = torch.cat(all_views)
        big_pdgs = torch.cat(all_hit_pdgs)
    else:
        big_coords = torch.tensor([])
        big_zs = torch.tensor([])
        big_energies = torch.tensor([])
        big_views = torch.tensor([])
        big_pdgs = torch.tensor([])

    num_records = len(all_record_hit_counts)

    def create_dense_tensor(data_list, shape_suffix):
        full_shape = (num_records,) + shape_suffix
        dense = torch.full(full_shape, float("nan"), dtype=torch.float32)
        for i, val in enumerate(data_list):
            if val is not None:
                dense[i] = val
        return dense

    dense_pion_stop = create_dense_tensor(all_record_true_pion_stops, (3,))
    dense_angle_vector = create_dense_tensor(all_record_true_angle_vectors, (3,))
    dense_group_probs = create_dense_tensor(all_record_group_probs, (3,))
    dense_pred_endpoints = create_dense_tensor(all_record_pred_endpoints, (2, 3, 3))

    data_dict = {
        "hit_coords": big_coords,
        "hit_zs": big_zs,
        "hit_energies": big_energies,
        "hit_views": big_views,
        "hit_pdgs": big_pdgs,
        "record_hit_counts": torch.tensor(all_record_hit_counts, dtype=torch.long),
        "record_event_ids": torch.tensor(all_record_event_ids, dtype=torch.long),
        "record_group_ids": torch.tensor(all_record_group_ids, dtype=torch.long),
        "record_true_pion_stop": dense_pion_stop,
        "record_true_angle_vector": dense_angle_vector,
        "record_group_probs": dense_group_probs,
        "record_pred_endpoints": dense_pred_endpoints,
        "event_record_counts": torch.tensor(event_record_counts, dtype=torch.long),
        "event_origins": torch.tensor(all_event_origins, dtype=torch.long),
        "num_events": len(events),
    }

    print(f"Saving to {path}...")
    torch.save(data_dict, path)
    print("Done.")


class MixedEventDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        print(f"Loading dataset from {path}...")
        self.data = torch.load(path, map_location="cpu")

        self.num_events = self.data["num_events"]

        self.hit_coords = self.data["hit_coords"]
        self.hit_zs = self.data["hit_zs"]
        self.hit_energies = self.data["hit_energies"]
        self.hit_views = self.data["hit_views"]
        self.hit_pdgs = self.data["hit_pdgs"]

        self.record_hit_counts = self.data["record_hit_counts"]
        self.record_event_ids = self.data["record_event_ids"]
        self.record_group_ids = self.data["record_group_ids"]
        self.record_true_pion_stop = self.data["record_true_pion_stop"]
        self.record_true_angle_vector = self.data["record_true_angle_vector"]

        self.record_group_probs = self.data.get("record_group_probs")
        self.record_pred_endpoints = self.data.get("record_pred_endpoints")

        self.event_record_counts = self.data["event_record_counts"]
        self.event_origins = self.data["event_origins"]

        self.event_offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.event_record_counts, dim=0)])
        self.record_offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.record_hit_counts, dim=0)])

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        start_record_idx = self.event_offsets[idx].item()
        end_record_idx = self.event_offsets[idx + 1].item()

        records: List[GraphRecord] = []
        origins: List[int] = []

        for r_idx in range(start_record_idx, end_record_idx):
            start_hit = self.record_offsets[r_idx].item()
            end_hit = self.record_offsets[r_idx + 1].item()

            origin = self.event_origins[r_idx].item()
            origins.append(origin)

            c = self.hit_coords[start_hit:end_hit]
            z = self.hit_zs[start_hit:end_hit]
            e = self.hit_energies[start_hit:end_hit]
            v = self.hit_views[start_hit:end_hit]
            p = self.hit_pdgs[start_hit:end_hit]

            def get_val(tensor_list, i):
                if tensor_list is None:
                    return None
                val = tensor_list[i]
                if torch.isnan(val.view(-1)[0]):
                    return None
                return val.numpy()

            pion_stop_val = get_val(self.record_true_pion_stop, r_idx)
            angle_val = get_val(self.record_true_angle_vector, r_idx)

            group_probs_val = get_val(self.record_group_probs, r_idx)
            if group_probs_val is not None:
                group_probs_val = group_probs_val.tolist()

            pred_endpoints_val = get_val(self.record_pred_endpoints, r_idx)
            pred_pion_stop_val = None
            if pred_endpoints_val is not None:
                pred_pion_stop_val = pred_endpoints_val[1].tolist()
                pred_endpoints_val = pred_endpoints_val.tolist()

            rec = GraphRecord(
                coord=c.numpy(),
                z=z.numpy(),
                energy=e.numpy(),
                view=v.numpy(),
                hit_pdgs=p.numpy(),
                event_id=self.record_event_ids[r_idx].item(),
                group_id=self.record_group_ids[r_idx].item(),
                true_pion_stop=pion_stop_val,
                true_angle_vector=angle_val,
                group_probs=group_probs_val,
                pred_endpoints=pred_endpoints_val,
                pred_pion_stop=pred_pion_stop_val,
            )
            records.append(rec)

        return EventContainer(records, origins)
