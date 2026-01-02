import torch
import torch.nn.functional as F
from typing import List, Optional, Any, Dict
from torch_geometric.data import Data, Batch

from pioneerml.data.datasets.graph_group import GraphRecord
from pioneerml.data.datasets.utils import fully_connected_edge_index, build_edge_attr
from pioneerml.models.classifiers import GroupClassifier
from pioneerml.models.regressors import OrthogonalEndpointRegressor
from pioneerml.data.event_mixer import EventContainer


class UpstreamPipeline:
    def __init__(self, device: torch.device):
        self.device = device
        self.models = {}
        self.classifier = GroupClassifier().to(self.device)
        self.endpoint_regressor = OrthogonalEndpointRegressor().to(self.device)

    def load_models(
        self,
        classifier_path: str,
        endpoint_path: str,
        classifier_config: Optional[Dict[str, Any]] = None,
        endpoint_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Loads the pre-trained models.
        :param classifier_config: Dictionary of arguments for GroupClassifier (e.g. {'hidden': 150})
        :param endpoint_config: Dictionary of arguments for OrthogonalEndpointRegressor
        """

        print(f"Loading classifier from {classifier_path}")
        cls_kwargs = classifier_config if classifier_config is not None else {}
        self.classifier = GroupClassifier(**cls_kwargs).to(self.device)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.classifier.train()

        print(f"Loading endpoint regressor from {endpoint_path}")
        end_kwargs = endpoint_config if endpoint_config is not None else {}
        self.endpoint_regressor = OrthogonalEndpointRegressor(**end_kwargs).to(self.device)
        self.endpoint_regressor.load_state_dict(torch.load(endpoint_path, map_location=self.device))
        self.endpoint_regressor.train()

    def _graph_record_to_data(self, record: GraphRecord) -> Data:
        """
        Converts GraphRecord to PyG Data object for inference.
        Features: [coord, z, energy, view] (4D)
        """
        coord = torch.tensor(record.coord, dtype=torch.float, device=self.device)
        z = torch.tensor(record.z, dtype=torch.float, device=self.device)
        energy = torch.tensor(record.energy, dtype=torch.float, device=self.device)
        view = torch.tensor(record.view, dtype=torch.float, device=self.device)

        node_features = torch.stack([coord, z, energy, view], dim=1)

        edge_index = fully_connected_edge_index(len(coord), device=self.device)
        edge_attr = build_edge_attr(node_features, edge_index)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        data.batch = torch.zeros(len(coord), dtype=torch.long, device=self.device)

        data.u = energy.sum().view(1, 1)

        return data

    def process_unmixed_events(self, events_list: List[List[GraphRecord]], batch_size: int = 200):
        """
        Runs the pipeline on raw unmixed events (List[List[GraphRecord]]) in-place.
        Updates each GraphRecord with group_probs and pred_endpoints.
        """
        all_records = [r for event in events_list for r in event]
        total = len(all_records)
        print(f"Processing {total} time groups from {len(events_list)} events in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch_records = all_records[i : i + batch_size]
            if not batch_records:
                continue

            data_list = [self._graph_record_to_data(r) for r in batch_records]
            batch = Batch.from_data_list(data_list).to(self.device)

            with torch.no_grad():
                logits = self.classifier(batch)
                probs_tensor = torch.sigmoid(logits)
                probs = probs_tensor.cpu().numpy()

                batch.group_probs = probs_tensor

                eps = self.endpoint_regressor(batch).reshape(-1, 2, 3, 3).cpu().numpy()

            for j, record in enumerate(batch_records):
                record.group_probs = probs[j].tolist()
                record.pred_endpoints = eps[j].tolist()
                record.pred_pion_stop = eps[j, 1, :, 1].tolist()

            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_records)}/{total} groups")
