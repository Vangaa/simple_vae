import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import utils
from autoencoder import Autoencoder

"""ffmpeg -i %03d.png output.gif -vf fps=0.1
"""

def main():
    args = utils.get_args()
    
    print("Prepare dataset...")
    mnist = input_data.read_data_sets("mnist/", one_hot = True)
    
    with tf.Graph().as_default(), tf.Session() as session:
        autoencoder = Autoencoder(
            784, args.hid_shape, args.lat_shape,
            optimizer = tf.train.AdagradOptimizer(args.lr),
            batch_size = args.batch_size,
            dropout = args.dropout)
        
        session.run(tf.initialize_all_variables())

        if args.save_model or args.load_model:
            saver = tf.train.Saver()

        if args.load_model:
            try:
                saver.restore(session, utils.SAVER_FILE)
            except ValueError:
                print("Cant find model file")
                sys.exit(1)
                
        if args.make_imgs:
            index = 0
            print("Prepare images directory...")
            utils.prepare_image_folder()
            example = utils.get_example(args.digit, mnist.test)
            
        print("Start training...")
        for epoch in range(args.epoches):
            for i, batch in enumerate(utils.gen_data(args.batch_size, mnist.train.images)):
                autoencoder.fit_on_batch(session, batch)
                if (i+1) % args.log_after == 0:
                    test_cost = autoencoder.evaluate(session, mnist.test.images)
                    print("Test error = {0:.4f} on {1} batch in {2} epoch".format(test_cost, i+1, epoch+1))
                    
                    if args.make_imgs:
                        path = os.path.join(utils.IMG_FOLDER, "{0:03}.png".format(index))
                        autoencoded = autoencoder.encode_decode(session, example.reshape(1, 784))
                        utils.save_image(autoencoded.reshape((28, 28)), path)
                        index += 1
            if args.save_model:
                saver.save(session, utils.SAVER_FILE)
                print("Model saved")

if __name__ == "__main__":
    main()