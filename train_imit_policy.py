import pickle, tensorflow as tf, tf_util, numpy as np
from tensorflow.contrib.labeled_tensor import batch

import imit_policy as imit

import parameter as par

n_h1 = par.n_h1
n_h2 = par.n_h2
n_h3 = par.n_h3
batch_size = par.batch_size
'''
def placeholder_inputs(size,nin,nout):
  x_placeholder = tf.placeholder(tf.float32, shape=(size,
                                                         nin))
  y_placeholder = tf.placeholder(tf.float32, shape=(size,
                                                         nout))
  return x_placeholder, y_placeholder

def fill_feed_dict(x, y, data, i, nin, nout):
    x_feed = data['observations'][i*batch_size:(i+1)*batch_size]
    y_feed = data['actions'][i*batch_size:(i+1)*batch_size].reshape(batch_size,nout)
    feed_dict = {
        x: x_feed,
        y: y_feed,
    }
    return feed_dict
'''
def train(args):
    with open('rollouts/'+args.name+'-expert', 'rb') as f:
        data = pickle.load(f)
    tempx=data['observations']
    temp=tempx.shape
    length=temp[0]
    nin=temp[1]
    tempy=data['actions']
    temp=tempy.shape
    nout=temp[2]
    total_batch = int(length/batch_size)

    print(nin)
    print(nout)

    x, y = imit.placeholder_inputs(None,nin,nout,batch_size)
    
    logits = imit.inference(x, nin, nout, n_h1, n_h2, n_h3)
    loss = imit.loss(logits, y)
    train_op = imit.training(loss,args.step_size)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    
    sess.run(init)
    if args.firsttime!=1:
        saver.restore(sess, "trainedNN/" + args.name)
    
    for epoch in range(args.num_epoches):
        avg_cost = 0
        for i in range(total_batch):
            feed_dict = imit.fill_feed_dict(x,y,data,i,nin,nout,batch_size)

            _, c =sess.run([train_op, loss], feed_dict=feed_dict)

            avg_cost += c / total_batch

        if epoch % 1 == 0:
            print("Epoch:","%04d"%(epoch+1), "cost=","{:.9f}".format(avg_cost))

    save_path = saver.save(sess, "trainedNN/"+args.name)
    sess.close()
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--num_epoches', type = int)
    parser.add_argument('--firsttime', type = int)
    parser.add_argument('--step_size', type = float)
    args = parser.parse_args()
    
    train(args)
