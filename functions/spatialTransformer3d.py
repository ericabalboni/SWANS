import tensorflow as tf
import numpy as np
from keras.engine.topology import Layer
from functions3d import createElipsoid2


class SpatialTransformer3D(Layer):
    #Trying to implement with a general transformation rather than an affine transformation

    #initalise the input deformation net. this needs to be in the form of a network produced Sequentially
    def __init__(self, def_net, output_size, **kwargs):
        self.defnet = def_net
        self.output_size = output_size
        super (SpatialTransformer3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.defnet.build(input_shape)                              #Build the deformation net
        self.trainable_weights = self.defnet.trainable_weights      #load the trainable weights, we learn this network end to end
        #self.trainable_weights.append(self.defnet.trainable_weights)

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None, int(output_size[0]), int(output_size[1]), int(output_size[2]), int(input_shape[-1]))   #last value is the number of channels

    def call(self, X, mask = None):
        #The transformation is the output of the defnet call
        transformation = self.defnet.call(X)
        #call the transformation function, all the other functions are called within this function
        #Output is the stack of transformed images
        output = self._transform(transformation, X, self.output_size)
        return output
    
    def get_config(self):     
        config = {
            'def_net': self.defnet,
            'output_size': self.output_size
        }
        base_config = super(SpatialTransformer3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype = 'int32')
        x = tf.reshape(x, shape = (-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, z, output_size):
        #Find the dimension information from the image input
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        depth = tf.shape(image)[3]
        num_channels = tf.shape(image)[4]

        #Turn x and y to floats
        x = tf.cast(x, dtype = 'float32')
        y = tf.cast(y, dtype = 'float32')
        z = tf.cast(z, dtype = 'float32')

        height_float = tf.cast(height, dtype = 'float32')
        width_float  = tf.cast(width,  dtype = 'float32')
        depth_float  = tf.cast(depth,  dtype = 'float32')

        output_height = output_size[0]
        output_width = output_size[1]
        output_depth = output_size[2]

        #find the current locations
        x = 0.5 * (x + 1.0) * (width_float)
        y = 0.5 * (y + 1.0) * (height_float)
        z = 0.5 * (z + 1.0) * (depth_float)

        #find the value before and after the current locations
        x0 = tf.cast(tf.floor(x), dtype = 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), dtype = 'int32')
        y1 = y0 + 1
        z0 = tf.cast(tf.floor(z), dtype = 'int32')
        z1 = z0 + 1

        #find the maximum values
        max_y = tf.cast(height - 1, dtype = 'int32')
        max_x = tf.cast(width - 1,  dtype = 'int32')
        max_z = tf.cast(depth - 1,  dtype = 'int32')
        zero = tf.zeros([], dtype = 'int32')

        #get rid of values which fall out of the range of pixel values
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        z0 = tf.clip_by_value(z0, zero, max_z)
        z1 = tf.clip_by_value(z1, zero, max_z)

        dim3 = depth
        dim2 = depth*width
        dim1 = depth*width*height
        base = self._repeat(tf.range(batch_size)*dim1,
                            output_height*output_width*output_depth)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0*dim3 + z0
        idx_b = base_y1 + x0*dim3 + z0
        idx_c = base_y0 + x1*dim3 + z0
        idx_d = base_y1 + x1*dim3 + z0
        idx_e = base_y0 + x0*dim3 + z1
        idx_f = base_y1 + x0*dim3 + z1
        idx_g = base_y0 + x1*dim3 + z1
        idx_h = base_y1 + x1*dim3 + z1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        #Create the map to sample from
        ellipse = createElipsoid2((32,64,64), scale=11, center=[8, 10, 10])
        ellipse = np.reshape(ellipse, (32, 64, 64, 1))
        ellipse = tf.convert_to_tensor(ellipse)


        mapp = tf.tile(ellipse, tf.stack([batch_size, 1, 1, 3]))
        """
        mapp=tf.reshape(ellipse, (1,32,64,64,1))
        mapp = tf.tile(ellipse, tf.stack([batch_size, 1, 1, 1]))
        """
        flat_image = tf.reshape(mapp, shape = (-1, num_channels))
        im_flat = tf.cast(flat_image, dtype = 'float32')

        '''
        im_flat = tf.reshape(image, tf.stack([-1, num_channels]))
        im_flat = tf.cast(im_flat, 'float32')
        '''
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)
        Ie = tf.gather(im_flat, idx_e)
        If = tf.gather(im_flat, idx_f)
        Ig = tf.gather(im_flat, idx_g)
        Ih = tf.gather(im_flat, idx_h)

        # and finally calculate interpolated values
        x1_f = tf.cast(x1, 'float32')
        y1_f = tf.cast(y1, 'float32')
        z1_f = tf.cast(z1, 'float32')

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        wa = tf.expand_dims((dz * dx * dy), 1)
        wb = tf.expand_dims((dz * dx * (1-dy)), 1)
        wc = tf.expand_dims((dz * (1-dx) * dy), 1)
        wd = tf.expand_dims((dz * (1-dx) * (1-dy)), 1)
        we = tf.expand_dims(((1-dz) * dx * dy), 1)
        wf = tf.expand_dims(((1-dz) * dx * (1-dy)), 1)
        wg = tf.expand_dims(((1-dz) * (1-dx) * dy), 1)
        wh = tf.expand_dims(((1-dz) * (1-dx) * (1-dy)), 1)

        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id,
                           we*Ie, wf*If, wg*Ig, wh*Ih])

        return output

    def _meshgrid(self, height, width, depth):
        #create a unit mesh covering the image
        x_linspace = tf.linspace(-1.0, 1.0, width)
        y_linspace = tf.linspace(-1.0, 1.0, height)
        z_linspace = tf.linspace(-1.0, 1.0, depth)
        x_coordinates, y_coordinates, z_coordinates = tf.meshgrid(x_linspace, y_linspace, z_linspace)
        return x_coordinates, y_coordinates, z_coordinates

    def _transform(self, transformation, input_shape, output_size):
        batch_size = tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[1]
        width = tf.shape(input_shape)[2]
        depth = tf.shape(input_shape)[3]
        num_channels = tf.shape(input_shape)[4]

        transformation = tf.cast(transformation, 'float32')
        transformation = tf.reshape(transformation, shape = (batch_size, height, width, depth, 3))

        output_height = output_size[0]
        output_width = output_size[1]
        output_depth = output_size[2]
        output_batchsize = output_size[3]

        indices_grid = self._meshgrid(output_height, output_width, output_depth)
        indices_grid = tf.expand_dims(indices_grid, 0)

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size, 1, 1, 1, 1]), name = 'tile1')
        indices_grid = tf.transpose(indices_grid, [0,2,3,4,1], name = 'indices_reshape')

        transformed_grid = tf.multiply(transformation, indices_grid, name = 'addition')

        x_s = tf.slice(transformed_grid, [0,0,0,0,0], [-1, -1, -1, -1, 1])
        y_s = tf.slice(transformed_grid, [0,0,0,0,1], [-1, -1, -1, -1, 1])
        z_s = tf.slice(transformed_grid, [0,0,0,0,2], [-1, -1, -1, -1, 1])

        x_s_flatten = tf.reshape(x_s, [-1], name ='xs_reshape')
        y_s_flatten = tf.reshape(y_s, [-1], name ='ys_reshape')
        z_s_flatten = tf.reshape(z_s, [-1], name ='zs_reshape')

        transformed_image = self._interpolate(input_shape, x_s_flatten, y_s_flatten, z_s_flatten, output_size)
        transformed_image = tf.reshape(transformed_image, shape = (batch_size, output_height, output_width, output_depth, num_channels))

        return transformed_image



