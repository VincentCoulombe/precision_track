from typing import Dict, Optional

import torch

from precision_track.utils import cosine_similarity


class AppearanceClassifier:
    def __init__(
        self,
        identities: Dict[int, str],
        features_size: int,
        device,
        precision,
        min_size_per_id: Optional[int] = 5,
        max_size_per_id: Optional[int] = 1000,
        confidence_threshold: Optional[float] = 0.8,
        variance_threshold: Optional[float] = 0.05,
        k: Optional[int] = 5,
    ):
        self.device = device
        self.identities = identities
        nb_identities = len(identities)
        self.nb_identities = nb_identities
        self._max_size_per_id = int(max_size_per_id)
        self._min_size_per_id = int(min_size_per_id)
        self._confidence_threshold = confidence_threshold
        self._variance_threshold = variance_threshold
        self._database_capacity = int(max_size_per_id) * nb_identities
        self._database = torch.zeros((self._database_capacity, int(features_size)), dtype=precision, device=device)
        self._identities = torch.zeros(self._database_capacity, dtype=torch.int64, device=device) - 1

        self._next_free_idx = 0
        self._identity_counts = torch.zeros(self.nb_identities, dtype=torch.int64, device=device)
        self._identity_next_slot = torch.zeros(self.nb_identities, dtype=torch.int64, device=device)  # Were to read next in the idx_map
        self._rolling_next_slot = torch.zeros(self.nb_identities, dtype=torch.int64, device=device)
        self._identities_idx_map = {identity: torch.zeros(max_size_per_id, dtype=torch.int64, device=device) for identity in identities}

        self._init_features = {identity: torch.zeros((min_size_per_id, int(features_size)), dtype=precision, device=device) for identity in identities}
        self._init_confidences = {identity: torch.zeros(min_size_per_id, dtype=precision, device=device) for identity in identities}
        self._is_initiated = torch.zeros(self.nb_identities, dtype=torch.bool, device=device)

        self._k = int(k)

    def __call__(self, features, identities, confidence_scores):
        similarities = self.get_similarities(features)
        for i, (feature, identity, score) in enumerate(zip(features, identities, confidence_scores)):
            identities[i] = self._confirm_identity(feature, identity, score, similarities[i])
        return identities

    def _confirm_identity(self, features, identity, score, similarities):
        identity = identity.item()
        assert isinstance(features, torch.Tensor)

        count = self._identity_counts[identity]
        next_slot = self._identity_next_slot[identity]

        if not self._is_initiated[identity]:
            init_features = self._init_features[identity]
            init_confidences = self._init_confidences[identity]

            init_features[count] = features
            init_confidences[count] = score

            self._identity_counts[identity] += 1

            if count == self._min_size_per_id:
                valid_samples = init_features
                valid_confidences = init_confidences

                # 1) Ensure that the track was reliable during the period
                all_confident = (valid_confidences > self._confidence_threshold).all()
                # 2) Ensure that the track's appearance remained stable during the period
                low_variance = torch.var(valid_samples, dim=0).mean() < self._variance_threshold

                if all_confident and low_variance:
                    for i in range(count):
                        if self._next_free_idx < self._database_capacity:
                            self._database[self._next_free_idx] = init_features[i]
                            self._identities[self._next_free_idx] = identity
                            self._identities_idx_map[identity][next_slot] = self._next_free_idx

                            self._next_free_idx += 1
                            next_slot += 1
                            self._identity_next_slot[identity] = next_slot
                            self._is_initiated[identity] = True
                else:
                    self._identity_counts[identity] = 0
                    self._is_initiated[identity] = False
            return identity
        else:
            k = min(self._identity_counts[identity], self._k)
            pred_identity, knn_count = self._knn_classification(k=k, similarities=similarities)
            knn_pred_is_confident = (knn_count / k) >= self._confidence_threshold
            all_identities_were_represented = torch.all(self._is_initiated)

            if pred_identity < 0 or not knn_pred_is_confident or not all_identities_were_represented:
                return identity

            # Enforce an equal representation of each identity
            equal_counts = (self._identity_counts[pred_identity] - self._identity_counts.min()) <= 1 and all_identities_were_represented

            pred_identity_int = pred_identity.item()

            if count < self._max_size_per_id and self._next_free_idx < self._database_capacity and equal_counts:
                # Append to database
                self._database[self._next_free_idx] = features
                self._identities[self._next_free_idx] = pred_identity
                self._identities_idx_map[pred_identity_int][next_slot] = self._next_free_idx

                self._next_free_idx += 1
                self._identity_counts[pred_identity] = count + 1
                self._identity_next_slot[pred_identity] = next_slot + 1
            else:
                # Add to database (Roll)
                rolling_next_slot = self._rolling_next_slot[pred_identity]
                rolling_next_slot = rolling_next_slot % self._identity_counts[pred_identity]
                db_idx = self._identities_idx_map[pred_identity_int][rolling_next_slot]
                self._database[db_idx] = features
                self._identities[db_idx] = pred_identity
                self._rolling_next_slot[pred_identity] = rolling_next_slot + 1

            return pred_identity

    def purge(self, identity_indices):
        for identity in identity_indices.tolist():
            count = self._identity_counts[identity].item()
            if count > 0:
                slots = self._identities_idx_map[identity][:count]
                self._identities[slots] = -1
                self._database[slots] = 0
            self._identity_counts[identity] = 0
            self._identity_next_slot[identity] = 0
            self._rolling_next_slot[identity] = 0
            self._is_initiated[identity] = False
            self._init_features[identity].zero_()
            self._init_confidences[identity].zero_()

    def get_similarities(self, features):
        similarities = cosine_similarity(features, self._database[: self._next_free_idx])
        return similarities

    def _knn_classification(self, k, similarities):
        _, topk_idxs = similarities.topk(k=k, dim=0)
        topk_identities = self._identities[topk_idxs]
        vals, counts = torch.unique(topk_identities, return_counts=True)
        max_count = counts.max()
        winners = vals[max_count == counts]
        if len(winners) == 1:
            best_pred = winners[0]
        else:
            # There is a tie, the classification will not count
            best_pred = -1
        return best_pred, max_count
