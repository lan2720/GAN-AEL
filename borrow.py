# coding:utf-8
import torch.nn as nn
import torch
import os
import sys
sys.path.append('..')
from encoder import EncoderRNN
from decoder import DecoderRNN, DecoderRNNWithGlobal
#from attn_decoder import AttnDecoderRNN

class Seq2Seq(nn.Module):
    def __init__(self, config, enc_birnn, dec_type):
        super(Seq2Seq, self).__init__()
        self.name = "seq2seq"
        self.config = config
        if config.vocab_size < 0:
            raise RuntimeError('Please specify --vocab_size')
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.encoder = EncoderRNN(config.embed_dim,
                                  config.hidden_dim,
                                  config.n_layers,
                                  config.dp_ratio,
                                  birnn=enc_birnn)
        if dec_type == "dec":
            self.decoder = DecoderRNN(config.embed_dim,
                                      self.encoder.hidden_size*self.encoder.num_directions,
                                      config.vocab_size,
                                      config.n_layers,
                                      config.dp_ratio)
        elif dec_type == 'dec_glo':
            self.decoder = DecoderRNNWithGlobal(config.embed_dim,
                                      self.encoder.hidden_size*self.encoder.num_directions,
                                      config.vocab_size,
                                      config.n_layers,
                                      config.dp_ratio)
        else:
            raise Exception("Unknown decoder type")
        if config.vocab_size < 0:
            raise RuntimeError('Please specify --vocab_size')


    def forward(self, enc_input, enc_input_lengths, dec_input):
        """
        enc_input = [batch_size, max_seq_len]
        enc_input_lengths是enc_input中每个example的真实长度，list是降序
        dec_inpit = [batch_size, max_seq_len]
        """
        encoder_input_embed = self.embed(enc_input) # [batch, time, emb_size]
        decoder_input_embed = self.embed(dec_input)
        enc_outputs, dec_init_state = self.encoder(encoder_input_embed, enc_input_lengths)
        # ****** NO ATTENTION HERE ******
        dec_logits, _ = self.decoder(decoder_input_embed, dec_init_state)
        return dec_logits

    def save(self, save_dir):
        if not os.path.exists(save_dir):
        #   raise RuntimeError("save_dir %s has existed, please remove first" % save_dir)
            os.mkdir(save_dir)
        else:
            if "enc_params.pkl" in os.listdir(save_dir) and self.config.continue_train == False:
                raise RuntimeError("Do you want to continue training model in %s?\nIf so, you should set `--continue`" % save_dir)
        torch.save(self.embed.state_dict(), os.path.join(save_dir, "w2v.pkl"))
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, 
                                "{}_params.pkl".format(self.encoder.name)))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir,
                                "{}_params.pkl".format(self.decoder.name)))
        self.config.continue_train = True

    def load_word_embed(self):
        w2v_path = os.path.join(self.config.save_dir, "w2v.pkl")
        print("%s load pretrained word2vec from %s" % (self.name, w2v_path))
        self.embed.load_state_dict(torch.load(w2v_path))

    def reload_all_params(self, save_dir):
        """
        reload全部参数，用于继续训练
        """
        if os.path.exists(save_dir) and "enc_params.pkl" in os.listdir(save_dir):
            self.load_word_embed()
            enc_path = os.path.join(save_dir, "%s_params.pkl" % self.encoder.name)
            dec_path = os.path.join(save_dir, "%s_params.pkl" % self.decoder.name)
            self.encoder.load_state_dict(torch.load(enc_path))
            self.decoder.load_state_dict(torch.load(dec_path))
            print("Reload all model params from %s" % save_dir)
        else:
            raise RuntimeError("No stored model to continue training in %s" % save_dir) 
