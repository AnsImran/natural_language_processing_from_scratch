import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import log_softmax

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

# import torchtext
# torchtext.disable_torchtext_deprecation_warning()

# import tensorflow as tf

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
# import pandas as pd

# import matplotlib.pyplot as plt
# import time
# import utils

# import textwrap
# wrapper = textwrap.TextWrapper(width=70)










def positional_encoding(positions, d_model):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int): Maximum number of positions to be encoded 
        d_model (int):   Encoding size 
    
    Returns:
        pos_encoding (torch.tensor): A matrix of shape (1, position, d_model) with the positional encodings
    """
    
    position = np.arange(positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2
    
    # initialize a matrix angle_rads of all the angles 
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads = position * angle_rates
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return torch.tensor(pos_encoding, dtype=torch.float32)



def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids (matrix like): matrix of size (n, m)
    
    Returns:
        mask : binary tensor of size (n, m)
    """
    # if decoder_token_ids.shape[1] <= 1:
    #     mask = None
    # else:
    mask = decoder_token_ids == 0
    return mask



# tensorflow implementation
def create_look_ahead_mask(matrix_of_sequences):
    """
    Returns a lower triangular matrix filled with ones
    
    Arguments:
        sequence_length (int): matrix size
    
    Returns:
        mask (torch.tensor): binary tensor of size (sequence_length, sequence_length)
    """

    # if matrix_of_sequences.shape[1] <= 1:
    #     mask = None
    # else:
    mask = torch.triu(torch.ones(matrix_of_sequences.shape[1], matrix_of_sequences.shape[1]), diagonal=1).to(torch.bool)
    return mask








class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim_, num_heads_, fully_connected_dim_,  dropout_rate_=0.1, layernorm_eps_=1e-6):
        super(EncoderLayer, self).__init__()
        
        self.attention           = nn.MultiheadAttention(embed_dim=embedding_dim_, num_heads=num_heads_, dropout=dropout_rate_, batch_first=True)
        self.layernorm1          = nn.BatchNorm1d(num_features=embedding_dim_, eps=layernorm_eps_)

        self.fc1                 = nn.Linear(in_features=embedding_dim_, out_features=fully_connected_dim_)
        self.fc2                 = nn.Linear(in_features=fully_connected_dim_, out_features=embedding_dim_)
        self.layernorm2          = nn.BatchNorm1d(num_features=embedding_dim_, eps=layernorm_eps_)

        self.dropout = nn.Dropout(dropout_rate_)

    def forward(self, x, mask):
        # x: (batch_size, seq_length, embedding_dim)
        # self attention
        attention_output, attention_weights = self.attention(query=x, key=x, value=x, key_padding_mask=mask)  # Self attention (batch_size, input_seq_len, embedding_dim)

        # skip_connection
        skip_x_attention = x + attention_output                   # (batch_size, seq_length, embedding_dim)
        skip_x_attention = skip_x_attention.permute(0, 2, 1)      # (batch_size, embedding_dim, input_seq_len)
        skip_x_attention = self.layernorm1(skip_x_attention)      # (batch_size, embedding_dim, input_seq_len)
        skip_x_attention = skip_x_attention.permute(0, 2, 1)      # (batch_size, input_seq_len, embedding_dim)

        # Dense / Fully connected layers / Feed forward neural network
        fc1_output = self.fc1(skip_x_attention)
        fc1_output = torch.relu(fc1_output)

        fc2_output = self.fc2(fc1_output)
        fc2_output = torch.relu(fc2_output)        

        # dropout
        fc2_output = self.dropout(fc2_output)

        # 2nd skip connection
        skip_attention_fc = skip_x_attention + fc2_output
        skip_attention_fc = skip_attention_fc.permute(0, 2, 1)
        skip_attention_fc = self.layernorm2(skip_attention_fc)
        skip_attention_fc = skip_attention_fc.permute(0, 2, 1)

        return skip_attention_fc




class Encoder(nn.Module):
    """
    The entire Encoder starts by passing the input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    encoder Layers
        
    """  
    def __init__(self, num_layers_, embedding_dim_, num_heads_, fully_connected_dim_, input_vocab_size_, maximum_position_encoding_, dropout_rate_=0.1, layernorm_eps_=1e-6):
        super(Encoder, self).__init__()

        self.embedding_dim = embedding_dim_
        self.num_layers    = num_layers_

        self.embedding    = nn.Embedding(num_embeddings=input_vocab_size_, embedding_dim=self.embedding_dim, padding_idx=0)
        self.pos_encoding = positional_encoding(maximum_position_encoding_, self.embedding_dim)
        

        self.enc_layers = nn.ModuleList([EncoderLayer(embedding_dim_=self.embedding_dim,
                                        num_heads_=num_heads_,
                                        fully_connected_dim_=fully_connected_dim_,
                                        dropout_rate_=dropout_rate_,
                                        layernorm_eps_=layernorm_eps_) 
                           for _ in range(self.num_layers)])

        
        
        self.dropout = nn.Dropout(dropout_rate_)

    
    
    def forward(self, current_batch_of_sequences):
        """
        Forward pass for the Encoder
        
        Arguments:
            current_batch_of_sequences (torch.tensor):    Tensor of shape (batch_size, seq_len)
        Returns:
            x (torch.tensor): Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        mask    = create_padding_mask(current_batch_of_sequences).to(current_batch_of_sequences.device)
        seq_len = current_batch_of_sequences.shape[1]
        
        # Pass input through the Embedding layer
        x = self.embedding(current_batch_of_sequences)  # (batch_size, input_seq_len, embedding_dim)

        
        # Scale embedding by multiplying it by the square root of the embedding dimension
        x = x * torch.sqrt( torch.tensor(self.embedding_dim, dtype=torch.float32).to(x.device) )
                                                                                            # x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        
        # Add the position encoding to embedding
        x += self.pos_encoding[:, :seq_len, :].to(x.device)

        # Pass the encoded embedding through a dropout layer
        # use `training=training`
        x = self.dropout(x)
        
        # Pass the output through the stack of encoding layers 
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x  # (batch_size, input_seq_len, embedding_dim)





# GRADED FUNCTION: DecoderLayer
class DecoderLayer(nn.Module):
    """
    The decoder layer is composed by two multi-head attention blocks, 
    one that takes the new input and uses self-attention, and the other 
    one that combines it with the output of the encoder, followed by a
    fully connected block. 
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)        
        self.mha2 = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True) 

        self.fc1  = nn.Linear(in_features=embedding_dim, out_features=fully_connected_dim)
        self.fc2  = nn.Linear(in_features=fully_connected_dim, out_features=embedding_dim)

        self.layernorm1 = nn.BatchNorm1d(num_features=embedding_dim, eps=layernorm_eps)
        self.layernorm2 = nn.BatchNorm1d(num_features=embedding_dim, eps=layernorm_eps)
        self.layernorm3 = nn.BatchNorm1d(num_features=embedding_dim, eps=layernorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)


    
    def forward(self, x, enc_output, look_ahead_mask, padding_mask_dec_query, padding_mask_enc_key):
        """
        Forward pass for the Decoder Layer
        
        Arguments:
            x (torch.tensor):               Tensor of shape (batch_size, target_seq_len, embedding_dim)
            enc_output (torch.tensor):      Tensor of shape(batch_size, input_seq_len, embedding_dim)
            look_ahead_mask (torch.tensor): Boolean mask for the target_input
            padding_mask (torch.tensor):    Boolean mask for the second multihead attention layer
        Returns:
            out3 (torch.tensor):                Tensor of shape (batch_size, target_seq_len, embedding_dim)
            attn_weights_block1 (torch.tensor): Tensor of shape (batch_size, ..............................
            attn_weights_block2 (torch.tensor): Tensor of shape (batch_size, ..............................
        """
        
        ### START CODE HERE ###
        # enc_output.shape == (batch_size, input_seq_len, fully_connected_dim) embeddin_dim
        
        # BLOCK 1
        # calculate self-attention and return attention scores as attn_weights_block1.
        # Dropout will be applied during training (~1 line).
        # if look_ahead_mask != None:
        mult_attn_out1, attn_weights_block1 = self.mha1(query=x, key=x, value=x,
                                                        is_causal        = True,               ##################################################
                                                        attn_mask        = look_ahead_mask,
                                                        key_padding_mask = padding_mask_dec_query)
        # else:
        #     mult_attn_out1, attn_weights_block1 = self.mha1(query=x, key=x, value=x)


        # apply layer normalization (layernorm1) to the sum of the attention output and the input (~1 line)
        # skip_connection
        Q1 = x + mult_attn_out1
        Q1 = Q1.permute(0, 2, 1)      # (batch_size, embedding_dim, input_seq_len)
        Q1 = self.layernorm1(Q1)      # (batch_size, embedding_dim, input_seq_len)
        Q1 = Q1.permute(0, 2, 1)      # (batch_size, input_seq_len, embedding_dim)



        # BLOCK 2
        # calculate self-attention using the Q from the first block and K and V from the encoder output. 
        # Dropout will be applied during training
        # Return attention scores as attn_weights_block2 (~1 line) 
        mult_attn_out2, attn_weights_block2 = self.mha2(query=Q1, key=enc_output, value=enc_output, key_padding_mask=padding_mask_enc_key)

        
        # apply layer normalization (layernorm2) to the sum of the attention output and the output of the first block (~1 line)
        mult_attn_out2 = Q1 + mult_attn_out2
        mult_attn_out2 = mult_attn_out2.permute(0, 2, 1)      # (batch_size, embedding_dim, input_seq_len)
        mult_attn_out2 = self.layernorm2(mult_attn_out2)      # (batch_size, embedding_dim, input_seq_len)
        mult_attn_out2 = mult_attn_out2.permute(0, 2, 1)      # (batch_size, input_seq_len, embedding_dim)

        
        #BLOCK 3
        # pass the output of the multi-head attention layer through a ffn        
        # Dense / Fully connected layers / Feed forward neural network
        ffn_output = self.fc1(mult_attn_out2)
        ffn_output = torch.relu(ffn_output)

        ffn_output = self.fc2(ffn_output)
        ffn_output = torch.relu(ffn_output)        

        # dropout
        ffn_output = self.dropout_ffn(ffn_output)

        
        # apply layer normalization (layernorm3) to the sum of the ffn output and the output of the second block        
        out3 = ffn_output + mult_attn_out2
        out3 = out3.permute(0, 2, 1)      # (batch_size, embedding_dim, input_seq_len)
        out3 = self.layernorm3(out3)      # (batch_size, embedding_dim, input_seq_len)
        out3 = out3.permute(0, 2, 1)      # (batch_size, input_seq_len, embedding_dim)
        
        ### END CODE HERE ###

        return out3, attn_weights_block1, attn_weights_block2






# GRADED FUNCTION: Decoder
class Decoder(nn.Module):
    """
    The entire Encoder starts by passing the target input to an embedding layer 
    and using positional encoding to then pass the output through a stack of
    decoder Layers
        
    """ 
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, target_vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=self.embedding_dim, padding_idx=0)
        
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)

        self.dec_layers = nn.ModuleList([DecoderLayer(embedding_dim=self.embedding_dim,
                                        num_heads=num_heads,
                                        fully_connected_dim=fully_connected_dim,
                                        dropout_rate=dropout_rate,
                                        layernorm_eps=layernorm_eps) 
                           for _ in range(self.num_layers)])
       
        self.dropout = nn.Dropout(dropout_rate)



    
    def forward(self, current_batch_of_sequences, enc_input, enc_output):
        """
        Forward  pass for the Decoder
        
        Arguments:
            x (torch.tensor):               Tensor of shape (batch_size, target_seq_len, embedding_dim)
            enc_output (torch.tensor):      Tensor of shape(batch_size, input_seq_len, embedding_dim)
            look_ahead_mask (torch.tensor): Boolean mask for the target_input
            padding_mask (torch.tensor):    Boolean mask for the second multihead attention layer
        Returns:
            x (torch.tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            attention_weights (dict[str: tf.Tensor]): Dictionary of tensors containing all the attention weights
                                each of shape Tensor of shape (batch_size, .....................................
        """
        padding_mask_dec_query = create_padding_mask(current_batch_of_sequences).to(current_batch_of_sequences.device)        
        padding_mask_enc_key   = create_padding_mask(enc_input).to(current_batch_of_sequences.device)
        look_ahead_mask        = create_look_ahead_mask(current_batch_of_sequences).to(current_batch_of_sequences.device)
        seq_len                = current_batch_of_sequences.shape[1]

        attention_weights = {}
        
        ### START CODE HERE ###
        # create word embeddings 
        x = self.embedding(current_batch_of_sequences)
        
        # scale embeddings by multiplying by the square root of their dimension
        x = x * torch.sqrt( torch.tensor(self.embedding_dim, dtype=torch.float32).to(x.device) )
        
        # add positional encodings to word embedding
        x += self.pos_encoding [:, :seq_len, :].to(x.device)

        # apply a dropout layer to x
        x = self.dropout(x)
        

        # use a for loop to pass x through a stack of decoder layers and update attention_weights (~4 lines total)
        for i in range(self.num_layers):
            # pass x and the encoder output through a stack of decoder layers and save the attention weights
            # of block 1 and 2 (~1 line)
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask_dec_query, padding_mask_enc_key)

            #update attention_weights dictionary with the attention weights of block 1 and block 2
            attention_weights['decoder_layer{}_block1_self_att'.format(i+1)]   = block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i+1)] = block2
        ### END CODE HERE ###
        
        # x.shape == (batch_size, target_seq_len, fully_connected_dim)
        return x, attention_weights





# GRADED FUNCTION: Transformer
class Transformer(nn.Module):
    """
    Complete transformer with an Encoder and a Decoder
    """
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size, 
               target_vocab_size, max_positional_encoding_input,
               max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers_=num_layers,
                               embedding_dim_=embedding_dim,
                               num_heads_=num_heads,
                               fully_connected_dim_=fully_connected_dim,
                               input_vocab_size_=input_vocab_size,
                               maximum_position_encoding_=max_positional_encoding_input,
                               dropout_rate_=dropout_rate,
                               layernorm_eps_=layernorm_eps)

        

        self.decoder = Decoder(num_layers=num_layers, 
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               target_vocab_size=target_vocab_size, 
                               maximum_position_encoding=max_positional_encoding_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)

        self.final_layer = nn.Linear(in_features=embedding_dim, out_features=target_vocab_size)

    
    def forward(self, input_sentence, output_sentence):
        """
        Forward pass for the entire Transformer
        Arguments:
            input_sentence (torch.tensor): Tensor of shape (batch_size, input_seq_len, embedding_dim)
                                          An array of the indexes of the words in the input sentence
            output_sentence (torch.tensor): Tensor of shape (batch_size, target_seq_len, embedding_dim)
                                          An array of the indexes of the words in the output sentence
        Returns:
            final_output (torch.tensor): The final output of the model
            attention_weights (dict[str: torch.tensor]): Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, .....................................................
        """
        ### START CODE HERE ###
        # call self.encoder with the appropriate arguments to get the encoder output
        enc_output = self.encoder(input_sentence)
        
        # call self.decoder with the appropriate arguments to get the decoder output
        # dec_output.shape == (batch_size, tar_seq_len, embedding_dim)
        dec_output, attention_weights = self.decoder(output_sentence, input_sentence, enc_output)
        
        # pass decoder output through a linear layer and log_softmax (~1 line)
        final_output = self.final_layer(dec_output)
        final_output = torch.nn.functional.log_softmax(final_output, dim=-1)
        ### END CODE HERE ###

        return final_output, attention_weights





# # Test your function!
# n_layers                       = 3
# emb_d                          = 34   
# n_heads                        = 17
# fully_connected_dim            = 8
# input_vocab_size               = 300
# target_vocab_size              = 350
# max_positional_encoding_input  = 12
# max_positional_encoding_target = 12

# model = Transformer(n_layers, 
#     emb_d, 
#     n_heads, 
#     fully_connected_dim, 
#     input_vocab_size, 
#     target_vocab_size, 
#     max_positional_encoding_input,
#     max_positional_encoding_target).to(device)


# # 0 is the padding value
# sentence_a = torch.from_numpy(np.array([[2, 3, 1, 3, 0, 0, 0]])).to(torch.int).to(device)
# sentence_b = torch.from_numpy(np.array([[1, 4, 0, 0, 0, 0]])).to(torch.int).to(device)


# test_summary, att_weights = model(
#     sentence_a,
#     sentence_b
# )

# print(f"Using num_layers={n_layers}, target_vocab_size={target_vocab_size} and num_heads={n_heads}:\n")
# print(f"sentence_a has shape:{sentence_a.shape}")
# print(f"sentence_b has shape:{sentence_b.shape}")

# print(f"\nOutput of transformer (summary) has shape:{test_summary.shape}\n")
# print("Attention weights:")
# for name, tensor in att_weights.items():
#     print(f"{name} has shape:{tensor.shape}")


# # expected output:
# # Using num_layers=3, target_vocab_size=350 and num_heads=17:

# # sentence_a has shape:torch.Size([1, 7])
# # sentence_b has shape:torch.Size([1, 6])

# # Output of transformer (summary) has shape:torch.Size([1, 6, 350])

# # Attention weights:
# # decoder_layer1_block1_self_att has shape:torch.Size([1, 6, 6])
# # decoder_layer1_block2_decenc_att has shape:torch.Size([1, 6, 7])
# # decoder_layer2_block1_self_att has shape:torch.Size([1, 6, 6])
# # decoder_layer2_block2_decenc_att has shape:torch.Size([1, 6, 7])
# # decoder_layer3_block1_self_att has shape:torch.Size([1, 6, 6])
# # decoder_layer3_block2_decenc_att has shape:torch.Size([1, 6, 7])


















# GRADED FUNCTION: next_word
def next_word(model, encoder_input, output):
    """
    Helper function for summarization that uses the model to predict just the next word.
    Arguments:
        encoder_input (torch.tensor): Input data to summarize
        output (torch.tensor): (incomplete) target (summary)
    Returns:
        predicted_id (tf.Tensor): The id of the predicted word
    """

    # Run the prediction of the next word with the transformer model
    predictions, attention_weights = model(encoder_input, output)
    ### END CODE HERE ###

    predictions  = predictions[: ,-1:, :]
    predicted_id = torch.argmax(predictions, axis=-1).to(torch.int32)
    
    return predicted_id
    