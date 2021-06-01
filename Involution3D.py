import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

class Involution3D( layers.Layer ) :
    def __init__( self, filters, kernel_size = [ 3, 3, 3 ], strides = 1, padding = 'SYMMETRIC', channels_per_group = 16, reduce_ratio = 1, kernel_initializer = 'glorot_uniform' ) :
        super( Involution3D, self ).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.strides = strides
        self.padding = padding
        self.channels_per_group = channels_per_group
        self.groups = filters // channels_per_group  # number of channel groups: within each group the same kernel is used
        self.reduce_ratio = reduce_ratio
        if self.padding == 'SYMMETRIC' :
            dx = kernel_size[ 0 ] // 2
            dy = kernel_size[ 1 ] // 2
            dz = kernel_size[ 2 ] // 2
            cx = 0
            if kernel_size[ 0 ] % 2 == 0 :  # even kernel size
                cx = -1
            cy = 0
            if kernel_size[ 1 ] % 2 == 0 :  # even kernel size
                cy = -1
            cz = 0
            if kernel_size[ 2 ] % 2 == 0 :  # even kernel size
                cz = -1
            self.paddings = tf.constant( [ [ 0, 0 ], [ dx, dx + cx ], [ dy, dy + cy ], [ dz, dz + cz ], [ 0, 0 ] ] )
        self.initial_mapping = layers.Conv3D( filters, 1, kernel_initializer = kernel_initializer )
        self.reduce_mapping = tf.keras.Sequential(
            [
                layers.Conv3D( filters // reduce_ratio, 1, kernel_initializer = kernel_initializer ),
                # layers.BatchNormalization(),
                # layers.LayerNormalization(),
                tfa.layers.GroupNormalization( groups = self.groups ),
                layers.Activation( 'relu' ),
            ]
        )
        self.span_mapping = layers.Conv3D( kernel_size[ 0 ] * kernel_size[ 1 ] * kernel_size[ 2 ] * self.groups, 1, kernel_initializer = kernel_initializer )
        if strides > 1 :
            self.o_mapping = layers.AveragePooling3D( strides )

    def get_config( self ) :  # this function is needed to save the layer when saving model or checkpoints
        config = super( Involution3D, self ).get_config()
        config.update( { 'filters' : self.filters } )
        config.update( { 'kernel_size' : self.kernel_size } )
        config.update( { 'strides' : self.strides } )
        config.update( { 'padding' : self.padding } )
        config.update( { 'channels_per_group' : self.channels_per_group } )
        config.update( { 'groups' : self.groups } )
        config.update( { 'reduce_ratio' : self.reduce_ratio } )
        config.update( { 'kernel_initializer' : self.kernel_initializer } )
        return config

    def call( self, x ) :
        weight = self.span_mapping( self.reduce_mapping( x if self.strides == 1 else self.o_mapping( x ) ) )
        # split groups into a separate dimension and add a unit dimension for multiplication with channels_per_group in the image expanded via broadcasting
        _, d, h, w, c = K.int_shape( x )  # get a tuple of image dimensions before padding
        weight = K.expand_dims( K.reshape( weight, ( -1, d, h, w, self.kernel_size[ 0 ] * self.kernel_size[ 1 ] * self.kernel_size[ 2 ], self.groups ) ), axis = 6 )
        if self.padding == 'SYMMETRIC' :
            x = tf.pad( x, self.paddings, 'SYMMETRIC' )
        out = tf.extract_volume_patches( input = x if c == self.filters else self.initial_mapping( x ),
                                        ksizes = [ 1, self.kernel_size[ 2 ], self.kernel_size[ 0 ], self.kernel_size[ 1 ], 1 ],
                                        strides = [ 1, self.strides, self.strides, self.strides, 1 ],
                                        padding = 'SAME' if self.padding == 'SAME' else 'VALID' )  # get kernel-sized image patches around each voxel, output size is b, d, h, w, k[0] * k[1] * k[2] * filters
        out = K.reshape( out, ( -1, d, h, w, self.kernel_size[ 0 ] * self.kernel_size[ 1 ] * self.kernel_size[ 2 ], self.groups, self.channels_per_group ) )  # split kernels and groups into separate dimensions
        out = K.sum( weight * out, axis = -3 )  # convolve flattened patches with the weights along the flattened kernel dimension, broadcast the unit weight dimension into channels_per_group
        out = K.reshape( out, ( -1, d, h, w, self.filters ) )
        return out