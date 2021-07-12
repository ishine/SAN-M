import torch
import torch.nn as nn

from model.decoder import Decoder
from model.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_targets):
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)                 
        pred, gold, *_ = self.decoder(padded_targets, encoder_padded_outputs, input_lengths) 
        return pred, gold


    def recognize(self, input, input_length, beam_size, nbest, max_decode_len, text_tokenizer=None, verbose=False):

        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],beam_size, nbest, max_decode_len, text_tokenizer,verbose=verbose)
        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.create_model_from_package(package)
        return model

    @classmethod
    def create_model_from_package(cls, package):
        encoder = Encoder(package['d_input'],
                          package['n_layers_enc'],
                          package['n_head'],
                          package['d_k'],
                          package['d_v'],
                          package['d_model'],
                          package['d_inner'],
                          dropout=package['dropout'],
                          pe_maxlen=package['pe_maxlen'])
        decoder = Decoder(package['sos_id'],
                          package['eos_id'],
                          package['vocab_size'],
                          package['d_word_vec'],
                          package['n_layers_dec'],
                          package['n_head'],
                          package['d_k'],
                          package['d_v'],
                          package['d_model'],
                          package['d_inner'],
                          dropout=package['dropout'],
                          tgt_emb_prj_weight_sharing=package['tgt_emb_prj_weight_sharing'],
                          pe_maxlen=package['pe_maxlen'],
                          )
        model = cls(encoder, decoder)
        model.load_state_dict(package['state_dict'])
        return model


    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # encoder
            'd_input': model.encoder.d_input,
            'n_layers_enc': model.encoder.n_layers,
            'n_head': model.encoder.n_head,
            'd_k': model.encoder.d_k,
            'd_v': model.encoder.d_v,
            'd_model': model.encoder.d_model,
            'd_inner': model.encoder.d_inner,
            'dropout': model.encoder.dropout_rate,
            'pe_maxlen': model.encoder.pe_maxlen,
            # decoder
            'sos_id': model.decoder.sos_id,
            'eos_id': model.decoder.eos_id,
            'vocab_size': model.decoder.n_tgt_vocab,
            'd_word_vec': model.decoder.d_word_vec,
            'n_layers_dec': model.decoder.n_layers,
            'tgt_emb_prj_weight_sharing': model.decoder.tgt_emb_prj_weight_sharing,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package
