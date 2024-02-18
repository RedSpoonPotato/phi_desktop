# based on paper: https://arxiv.org/pdf/1810.04805.pdf
# inspiried heavily by HuggingFace's BertModel:
# https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/bert/modeling_bert.py#L865
import tensorflow as tf
import math

class Dense(tf.Module):
    def __init__(self, in_features:int, out_features:int, weights:tf.Tensor, 
                 is_biased=False, bias:tf.Tensor=None, name=None):
        super().__init__(name=name)
        if is_biased and bias is None:
            raise ValueError("is_biased == True, and yet no bias tensor provided")
        if not is_biased:
            bias = tf.zeros([out_features])
        self.in_features = in_features
        self.out_features = out_features
        self.w = tf.reshape(weights, (out_features, in_features))
        self.b = tf.reshape(bias, (1, out_features))
    @tf.function
    def __call__(self, x:tf.Tensor):
        out = tf.matmul(self.w , tf.transpose(x)) # out x 512
        return tf.transpose(out) + self.b # 512 x out

class Dense_v2(tf.Module):
    def __init__(self, in_features:int, out_features:int, weights:tf.Tensor, bias:tf.Tensor=None, name=None):
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.w = tf.reshape(weights, (in_features, out_features))
        if bias is None:
            self.b = tf.zeros([1, out_features])
        else:
            self.b = tf.reshape(bias, (1, out_features))
    @tf.function
    def __call__(self, x:tf.Tensor):
        out = tf.matmul(x, self.w)
        return out + self.b

class SDP_Multi_Attention(tf.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float, 
                 in_seq_len:int, out_seq_len:int, weights:dict, name=None):
        super().__init__(name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_k = d_model
        self.in_seq_len = in_seq_len # keys and values
        self.out_seq_len = out_seq_len # queries
        self.W_q = Dense(d_model, d_model, weights['attention.self.query.weight'], 
                         is_biased=True, bias=weights['attention.self.query.bias'])
        self.W_k = Dense(d_model, d_model, weights['attention.self.key.weight'], 
                         is_biased=True, bias=weights['attention.self.key.bias'])
        self.W_v = Dense(d_model, d_model, weights['attention.self.value.weight'], 
                        is_biased=True, bias=weights['attention.self.value.bias'])
        self.W_o = Dense(d_model, d_model, weights['attention.output.dense.weight'], 
                         is_biased=True, bias=weights['attention.output.dense.bias'])
    @tf.function
    def __call__(self, query, key, value, mask=None):
        # q,k,v: (512 x 768), mask: (512 x 512)
        features_per_head = self.d_model // self.num_heads
        assert(features_per_head * self.num_heads == self.d_model)
        q = self.W_q(query) # [512 x 768]
        k = self.W_k(key)
        v = self.W_v(value)
        q = tf.reshape(q, (self.out_seq_len, self.num_heads, features_per_head)) # [512 x 12 x 64]
        k = tf.reshape(k, (self.in_seq_len, self.num_heads, features_per_head))
        v = tf.reshape(v, (self.in_seq_len, self.num_heads, features_per_head))
        q = tf.transpose(q, perm=(1,0,2))
        k = tf.transpose(k, perm=(1,0,2))
        v = tf.transpose(v, perm=(1,0,2)) # heads, seq_len, features
        atten_matrix = tf.matmul(q, tf.transpose(k, perm=(0,2,1)))
        atten_matrix = tf.multiply(atten_matrix, 1.0 / math.sqrt(float(features_per_head)))
        if mask is not None:
            mask = tf.expand_dims(mask, axis=[0])
            adder = (1.0 - tf.cast(mask, tf.float32)) * -10000.0
            atten_matrix += adder
        atten_probs = tf.nn.softmax(atten_matrix)
         # (12, 512, 512i) x (12, 512i, 64) = (12, 512, 64)
        context_matrix = tf.matmul(atten_probs, v)
        context_matrix = tf.transpose(context_matrix, perm=(1, 0, 2))
        output = tf.reshape(context_matrix, (self.out_seq_len, self.d_model))
        # output is 2d
        output = self.W_o(output)
        return output

class LayerNorm(tf.Module):
    def __init__(self, weights, biases, eps=1e-12, name=None):
        super().__init__(name)
        # w's and b's are both 1d (768)
        self.weights = weights
        self.biases = biases
        self.eps = eps
    @tf.function
    def __call__(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) / tf.sqrt(variance + self.eps)
        x = self.weights * x + self.biases
        return x

class EncoderBlock(tf.Module):
    def __init__(self, d_model:int, seq_len:int, d_hidden:int, num_heads:int, dropout:float, weights:dict, 
                 name=None):
        super().__init__(name)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.seq_len = seq_len
        self.W_ff_1 = Dense(d_model, d_hidden, weights['intermediate.dense.weight'], 
                            is_biased=True, bias=weights['intermediate.dense.bias'])
        self.W_ff_2 = Dense(d_hidden, d_model, weights['output.dense.weight'],
                            is_biased=True, bias=weights['output.dense.bias'])
        self.attention = SDP_Multi_Attention(d_model, num_heads, dropout, seq_len, seq_len, weights)
        self.norm1 = LayerNorm(weights['attention.output.LayerNorm.weight'], weights['attention.output.LayerNorm.bias'])
        self.norm2 = LayerNorm(weights['output.LayerNorm.weight'], weights['output.LayerNorm.bias'])
    @tf.function
    def __call__(self, input, mask):
        # assume that input is 2d (batch is 1) 512 x 768
        out = self.norm1(input + self.attention(input, input, input, mask))
        out = self.norm2(out + self.W_ff_2(tf.nn.gelu(self.W_ff_1(out))))
        return out

class Encoder(tf.Module):
    def __init__(self, num_blocks:int, d_model:int, seq_len:int, d_hidden:int, num_heads:int, dropout:float, 
                 list_of_weights:list, name=None):
        super().__init__(name)
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.seq_len = seq_len
        self.encoder_list = [EncoderBlock(d_model, seq_len, d_hidden, num_heads, dropout, list_of_weights[i])
                        for i in range(num_blocks)]
    @tf.function
    def __call__(self, input:tf.Tensor, mask:tf.Tensor):
        # input is 2d, mask is 2d
        # MIGHT NOT BE ABLE TO DO FOR LOOPS IN GRAPH-MODE
        input = tf.reshape(input, [self.seq_len, self.d_model])
        mask = tf.reshape(mask, [1, self.seq_len])
        out = input
        mask_2d = tf.matmul(tf.transpose(mask), mask) # (512, 512)
        for encoder in self.encoder_list:
            out = encoder(out, mask_2d)
        return out

class Embedding_1d(tf.Module):
    def __init__(self, seq_len:int, d_model:int, embed_values:tf.Tensor, name=None):
        super().__init__(name)
        self.d_model = d_model
        self.seq_len = seq_len
        self.matrix = embed_values # 512 x 768
    @tf.function
    def __call__(self, indexes:tf.Tensor):
        # indexes is a 1d vector of ints, will return a 2d matrix
        output = tf.strided_slice(self.matrix, (indexes[0],0), (indexes[0]+1,self.d_model), (1,1))
        for i in range(1, len(indexes)):
            slice = tf.strided_slice(self.matrix, (indexes[i],0), (indexes[i]+1,self.d_model), (1,1))
            output = tf.concat([output, slice], 0)
        return output

class Embedding(tf.Module):
    def __init__(self, seq_len:int, d_model:int, embed_values:tf.Tensor, name=None):
        super().__init__(name)
        self.d_model = d_model
        self.seq_len = seq_len
        self.matrix = embed_values # 512 x 768
    @tf.function
    def __call__(self, indexes:tf.Tensor):
        # assume indexes.shape is 2d (batch_size, seq_len)
        output_3d = tf.zeros([1, indexes.shape[1], self.d_model], dtype=tf.float16) # a dummy 3d slice to keep code neat (b/c of tf.concat)
        for batch in range(indexes.shape[0]):
          output_2d = tf.strided_slice(self.matrix, (indexes[batch][0],0), (indexes[batch][0]+1,self.d_model), (1,1))
          for i in range(1, indexes.shape[1]):
              slice = tf.strided_slice(self.matrix, (indexes[batch][i],0), (indexes[batch][i]+1,self.d_model), (1,1))
              output_2d = tf.concat([output_2d, slice], 0)
          output_2d = tf.expand_dims(output_2d, axis=0)
          output_3d = tf.concat([output_3d, output_2d], 0)
        return output_3d[1:]
        
class BertEmbedding(tf.Module):
    def __init__(self, d_model:int, seq_len:int, vocab_size:int, max_seq_len:int, embeddings:dict, name=None):
        super().__init__(name)
        self.d_model = d_model
        self.seq_len = seq_len
        self.word_emb = Embedding(vocab_size, d_model, embeddings['embeddings.word_embeddings.weight'])
        self.pos_emb = Embedding(max_seq_len, d_model, embeddings['embeddings.position_embeddings.weight'])
        self.tok_emb = Embedding(2, d_model, embeddings['embeddings.token_type_embeddings.weight'])
        self.embed_norm = LayerNorm(embeddings['embeddings.LayerNorm.weight'], embeddings['embeddings.LayerNorm.bias']) 
    @tf.function
    def __call__(self, features:tf.Tensor, segments:tf.Tensor):
        # inputs are 1d
        pos = tf.range(self.seq_len)
        embeds = self.word_emb(features) + self.pos_emb(pos) + self.tok_emb(segments)
        return self.embed_norm(embeds)

class Bert(tf.Module):
    def __init__(self, num_blocks:int, d_model:int, seq_len:int, d_hidden:int, num_heads:int, dropout:float, 
                 vocab_size:int, max_seq_len:int, list_of_encoder_weights:list, embeddings:dict, name=None):
        super().__init__(name)
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.seq_len = seq_len
        self.embeds = BertEmbedding(d_model, seq_len, vocab_size, max_seq_len, embeddings)
        self.encoder = Encoder(num_blocks, d_model, seq_len, d_hidden, num_heads, dropout, list_of_encoder_weights)
    @tf.function(input_signature=[tf.TensorSpec(shape=[1,1,1,512], dtype=tf.float32), 
                                  tf.TensorSpec(shape=[1,1,1,512], dtype=tf.float32), 
                                  tf.TensorSpec(shape=[1,1,1,512], dtype=tf.float32)])
    def __call__(self, features:tf.Tensor, segments:tf.Tensor, mask:tf.Tensor):
        features = tf.reshape(features, (self.seq_len,))
        segments = tf.reshape(segments, (self.seq_len,))
        mask = tf.reshape(mask, (1, self.seq_len))
        features = tf.cast(features, dtype='int32')
        segments = tf.cast(segments, dtype='int32')
        out = self.embeds(features, segments) # 2d
        out = self.encoder(out, mask)
        return {
            'Bert_Output': out
        }

####################################################################################################################   
# Additional Wrappers for generating encoder_1.dlc and encoder_2.dlc

class Encoder_1(tf.Module):
    def __init__(self, num_blocks:int, d_model:int, seq_len:int, d_hidden:int, num_heads:int, dropout:float, 
                 list_of_encoder_weights:list, name=None):
        super().__init__(name)
        self.encoder = Encoder(num_blocks, d_model, seq_len, d_hidden, num_heads, dropout, list_of_encoder_weights)
        self.seq_len = seq_len
        self.d_model = d_model
    @tf.function(input_signature=[tf.TensorSpec(shape=[1,1,512,768], dtype=tf.float32), 
                                  tf.TensorSpec(shape=[1,1,1,512], dtype=tf.float32)])
    def __call__(self, input:tf.Tensor, mask:tf.Tensor):
        input = tf.reshape(input, [self.seq_len, self.d_model])
        mask = tf.reshape(mask, [1, self.seq_len])
        return {
            "Encoder_1": self.encoder(input, mask)
        }

# be sure to use the correct weights for Encoder_2 (list_of_weights[6:])
class Encoder_2(tf.Module):
    def __init__(self, num_blocks:int, d_model:int, seq_len:int, d_hidden:int, num_heads:int, dropout:float, 
                 list_of_encoder_weights:list, name=None):
        self.encoder = Encoder(num_blocks, d_model, seq_len, d_hidden, num_heads, dropout, list_of_encoder_weights)
        self.seq_len = seq_len
        self.d_model = d_model
    @tf.function(input_signature=[tf.TensorSpec(shape=[1,1,512,768], dtype=tf.float32), 
                                  tf.TensorSpec(shape=[1,1,1,512], dtype=tf.float32)])
    def __call__(self, input:tf.Tensor, mask:tf.Tensor):
        input = tf.reshape(input, [self.seq_len, self.d_model])
        mask = tf.reshape(mask, [1, self.seq_len])
        return {
            "Encoder_2": self.encoder(input, mask)
        }