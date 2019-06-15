from typing import Dict

from torch import tensor, cat, max, clamp, mean

from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.activations import Activation
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.similarity_functions.cosine import CosineSimilarity
from allennlp.data.vocabulary import Vocabulary

class CitationRanker(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 title_embedder: TextFieldEmbedder,
                 abstract_embedder: TextFieldEmbedder,
                 dense_dim=75) -> None:
        
        super().__init__(vocab)
        
        self.title_embedder = title_embedder
        self.abstract_embedder = abstract_embedder
        self.intermediate_dim = 6
        self.n_layers = 3
        self.layer_dims = [dense_dim for i in range(self.n_layers-1)]
        self.layer_dims.append(1)
        
        self.activations = [Activation.by_name("elu")(),Activation.by_name("elu")(),Activation.by_name("sigmoid")()]
        self.layers = FeedForward(input_dim=self.intermediate_dim,
                                  num_layers=self.n_layers,
                                  hidden_dims=self.layer_dims,
                                  activations=self.activations)
    
    def forward(self,
                query_title: Dict[str, tensor],
                query_abstract: Dict[str, tensor],
                candidate_title: Dict[str, tensor],
                candidate_abstract: Dict[str, tensor],
                candidate_citations: tensor,
                title_intersection: tensor,
                abstract_intersection: tensor,
                cos_sim: tensor,
                label: tensor = None) -> Dict[str,tensor]:
        
        query_title_embed = self.title_embedder.forward(query_title["tokens"])
        query_abstract_embed = self.abstract_embedder.forward(query_abstract["tokens"])
        
        candidate_title_embed = self.title_embedder.forward(candidate_title["tokens"])
        candidate_abstract_embed = self.abstract_embedder.forward(candidate_title["tokens"])
        
        title_cos_sim = CosineSimilarity().forward(query_title_embed,candidate_title_embed).unsqueeze(-1)
        
        abstract_cos_sim = CosineSimilarity().forward(query_abstract_embed,candidate_abstract_embed).unsqueeze(-1)
        
        intermediate_output = cat((title_cos_sim, abstract_cos_sim, candidate_citations, title_intersection, abstract_intersection, cos_sim), dim = -1)
        
        pred = self.layers.forward(intermediate_output)
        
        output = {"cite_prob":pred}
        
        if label is not None:
            output["loss"] = self._compute_loss(pred,label)
            
        return output
            
        
    def _compute_loss(self, pred: tensor, label: tensor) -> tensor:
        #in existing implementation training examples with even indices are positive/odd indices are negative
        positive = pred[::2]
        negative = pred[1::2]
        
        #"margin is given by the difference in label"
        margin = label[::2] - label[1::2]
        delta = clamp(margin + negative - positive,min=0)
        
        return mean(delta)
