??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8֗
?
cnn_policy/conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namecnn_policy/conv2d_14/kernel
?
/cnn_policy/conv2d_14/kernel/Read/ReadVariableOpReadVariableOpcnn_policy/conv2d_14/kernel*&
_output_shapes
: *
dtype0
?
cnn_policy/conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namecnn_policy/conv2d_14/bias
?
-cnn_policy/conv2d_14/bias/Read/ReadVariableOpReadVariableOpcnn_policy/conv2d_14/bias*
_output_shapes
: *
dtype0
?
cnn_policy/conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*,
shared_namecnn_policy/conv2d_15/kernel
?
/cnn_policy/conv2d_15/kernel/Read/ReadVariableOpReadVariableOpcnn_policy/conv2d_15/kernel*&
_output_shapes
: @*
dtype0
?
cnn_policy/conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namecnn_policy/conv2d_15/bias
?
-cnn_policy/conv2d_15/bias/Read/ReadVariableOpReadVariableOpcnn_policy/conv2d_15/bias*
_output_shapes
:@*
dtype0
?
cnn_policy/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_namecnn_policy/dense_1/kernel
?
-cnn_policy/dense_1/kernel/Read/ReadVariableOpReadVariableOpcnn_policy/dense_1/kernel* 
_output_shapes
:
??*
dtype0
?
cnn_policy/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namecnn_policy/dense_1/bias

+cnn_policy/dense_1/bias/Read/ReadVariableOpReadVariableOpcnn_policy/dense_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
z
c1
c2
f3
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

	kernel

bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*
	0

1
2
3
4
5
*
	0

1
2
3
4
5
 
?
layer_metrics
metrics
	variables
layer_regularization_losses
trainable_variables
regularization_losses

layers
non_trainable_variables
 
US
VARIABLE_VALUEcnn_policy/conv2d_14/kernel$c1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcnn_policy/conv2d_14/bias"c1/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1

	0

1
 
?
 non_trainable_variables
!layer_metrics
"metrics
	variables
#layer_regularization_losses
trainable_variables
regularization_losses

$layers
US
VARIABLE_VALUEcnn_policy/conv2d_15/kernel$c2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcnn_policy/conv2d_15/bias"c2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
%non_trainable_variables
&layer_metrics
'metrics
	variables
(layer_regularization_losses
trainable_variables
regularization_losses

)layers
SQ
VARIABLE_VALUEcnn_policy/dense_1/kernel$f3/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcnn_policy/dense_1/bias"f3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
*non_trainable_variables
+layer_metrics
,metrics
	variables
-layer_regularization_losses
trainable_variables
regularization_losses

.layers
 
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????``*
dtype0*$
shape:?????????``
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cnn_policy/conv2d_14/kernelcnn_policy/conv2d_14/biascnn_policy/conv2d_15/kernelcnn_policy/conv2d_15/biascnn_policy/dense_1/kernelcnn_policy/dense_1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_2372
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/cnn_policy/conv2d_14/kernel/Read/ReadVariableOp-cnn_policy/conv2d_14/bias/Read/ReadVariableOp/cnn_policy/conv2d_15/kernel/Read/ReadVariableOp-cnn_policy/conv2d_15/bias/Read/ReadVariableOp-cnn_policy/dense_1/kernel/Read/ReadVariableOp+cnn_policy/dense_1/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_2798
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecnn_policy/conv2d_14/kernelcnn_policy/conv2d_14/biascnn_policy/conv2d_15/kernelcnn_policy/conv2d_15/biascnn_policy/dense_1/kernelcnn_policy/dense_1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_2826??
?H
?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2466
input_1,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2Dinput_1'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ *
paddingVALID*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ 2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^^ 2
conv2d_14/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d_14/Relu:activations:0*/
_output_shapes
:?????????// *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dmax_pooling2d/MaxPool:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@*
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????--@2
conv2d_15/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? y  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_14/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.cnn_policy/conv2d_14/kernel/Regularizer/Square?
-cnn_policy/conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_14/kernel/Regularizer/Const?
+cnn_policy/conv2d_14/kernel/Regularizer/SumSum2cnn_policy/conv2d_14/kernel/Regularizer/Square:y:06cnn_policy/conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/Sum?
-cnn_policy/conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_14/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_14/kernel/Regularizer/mulMul6cnn_policy/conv2d_14/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/mul?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_15/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.cnn_policy/conv2d_15/kernel/Regularizer/Square?
-cnn_policy/conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_15/kernel/Regularizer/Const?
+cnn_policy/conv2d_15/kernel/Regularizer/SumSum2cnn_policy/conv2d_15/kernel/Regularizer/Square:y:06cnn_policy/conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/Sum?
-cnn_policy/conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_15/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_15/kernel/Regularizer/mulMul6cnn_policy/conv2d_15/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/mul?
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
,cnn_policy/dense_1/kernel/Regularizer/SquareSquareCcnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,cnn_policy/dense_1/kernel/Regularizer/Square?
+cnn_policy/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+cnn_policy/dense_1/kernel/Regularizer/Const?
)cnn_policy/dense_1/kernel/Regularizer/SumSum0cnn_policy/dense_1/kernel/Regularizer/Square:y:04cnn_policy/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/Sum?
+cnn_policy/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+cnn_policy/dense_1/kernel/Regularizer/mul/x?
)cnn_policy/dense_1/kernel/Regularizer/mulMul4cnn_policy/dense_1/kernel/Regularizer/mul/x:output:02cnn_policy/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/mul?
IdentityIdentitydense_1/Softmax:softmax:0>^cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp>^cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp<^cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::2~
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp2~
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp2z
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?
?
C__inference_conv2d_14_layer_call_and_return_conditional_losses_2651

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????^^ 2
Relu?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_14/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.cnn_policy/conv2d_14/kernel/Regularizer/Square?
-cnn_policy/conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_14/kernel/Regularizer/Const?
+cnn_policy/conv2d_14/kernel/Regularizer/SumSum2cnn_policy/conv2d_14/kernel/Regularizer/Square:y:06cnn_policy/conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/Sum?
-cnn_policy/conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_14/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_14/kernel/Regularizer/mulMul6cnn_policy/conv2d_14/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????^^ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????``::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????``
 
_user_specified_nameinputs
?
?
A__inference_dense_1_layer_call_and_return_conditional_losses_2142

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
,cnn_policy/dense_1/kernel/Regularizer/SquareSquareCcnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,cnn_policy/dense_1/kernel/Regularizer/Square?
+cnn_policy/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+cnn_policy/dense_1/kernel/Regularizer/Const?
)cnn_policy/dense_1/kernel/Regularizer/SumSum0cnn_policy/dense_1/kernel/Regularizer/Square:y:04cnn_policy/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/Sum?
+cnn_policy/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+cnn_policy/dense_1/kernel/Regularizer/mul/x?
)cnn_policy/dense_1/kernel/Regularizer/mulMul4cnn_policy/dense_1/kernel/Regularizer/mul/x:output:02cnn_policy/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/mul?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp<^cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2z
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_15_layer_call_and_return_conditional_losses_2683

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????--@2
Relu?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_15/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.cnn_policy/conv2d_15/kernel/Regularizer/Square?
-cnn_policy/conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_15/kernel/Regularizer/Const?
+cnn_policy/conv2d_15/kernel/Regularizer/SumSum2cnn_policy/conv2d_15/kernel/Regularizer/Square:y:06cnn_policy/conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/Sum?
-cnn_policy/conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_15/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_15/kernel/Regularizer/mulMul6cnn_policy/conv2d_15/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????--@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????// ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????// 
 
_user_specified_nameinputs
?
?
A__inference_dense_1_layer_call_and_return_conditional_losses_2715

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
,cnn_policy/dense_1/kernel/Regularizer/SquareSquareCcnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,cnn_policy/dense_1/kernel/Regularizer/Square?
+cnn_policy/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+cnn_policy/dense_1/kernel/Regularizer/Const?
)cnn_policy/dense_1/kernel/Regularizer/SumSum0cnn_policy/dense_1/kernel/Regularizer/Square:y:04cnn_policy/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/Sum?
+cnn_policy/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+cnn_policy/dense_1/kernel/Regularizer/mul/x?
)cnn_policy/dense_1/kernel/Regularizer/mulMul4cnn_policy/dense_1/kernel/Regularizer/mul/x:output:02cnn_policy/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/mul?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp<^cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2z
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
{
&__inference_dense_1_layer_call_fn_2724

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_21422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?H
?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2547	
input,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2Dinput'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ *
paddingVALID*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ 2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^^ 2
conv2d_14/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d_14/Relu:activations:0*/
_output_shapes
:?????????// *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dmax_pooling2d/MaxPool:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@*
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????--@2
conv2d_15/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? y  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_14/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.cnn_policy/conv2d_14/kernel/Regularizer/Square?
-cnn_policy/conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_14/kernel/Regularizer/Const?
+cnn_policy/conv2d_14/kernel/Regularizer/SumSum2cnn_policy/conv2d_14/kernel/Regularizer/Square:y:06cnn_policy/conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/Sum?
-cnn_policy/conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_14/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_14/kernel/Regularizer/mulMul6cnn_policy/conv2d_14/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/mul?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_15/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.cnn_policy/conv2d_15/kernel/Regularizer/Square?
-cnn_policy/conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_15/kernel/Regularizer/Const?
+cnn_policy/conv2d_15/kernel/Regularizer/SumSum2cnn_policy/conv2d_15/kernel/Regularizer/Square:y:06cnn_policy/conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/Sum?
-cnn_policy/conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_15/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_15/kernel/Regularizer/mulMul6cnn_policy/conv2d_15/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/mul?
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
,cnn_policy/dense_1/kernel/Regularizer/SquareSquareCcnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,cnn_policy/dense_1/kernel/Regularizer/Square?
+cnn_policy/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+cnn_policy/dense_1/kernel/Regularizer/Const?
)cnn_policy/dense_1/kernel/Regularizer/SumSum0cnn_policy/dense_1/kernel/Regularizer/Square:y:04cnn_policy/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/Sum?
+cnn_policy/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+cnn_policy/dense_1/kernel/Regularizer/mul/x?
)cnn_policy/dense_1/kernel/Regularizer/mulMul4cnn_policy/dense_1/kernel/Regularizer/mul/x:output:02cnn_policy/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/mul?
IdentityIdentitydense_1/Softmax:softmax:0>^cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp>^cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp<^cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::2~
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp2~
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp2z
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?
?
__inference_loss_fn_2_2757H
Dcnn_policy_dense_1_kernel_regularizer_square_readvariableop_resource
identity??;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDcnn_policy_dense_1_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
,cnn_policy/dense_1/kernel/Regularizer/SquareSquareCcnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,cnn_policy/dense_1/kernel/Regularizer/Square?
+cnn_policy/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+cnn_policy/dense_1/kernel/Regularizer/Const?
)cnn_policy/dense_1/kernel/Regularizer/SumSum0cnn_policy/dense_1/kernel/Regularizer/Square:y:04cnn_policy/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/Sum?
+cnn_policy/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+cnn_policy/dense_1/kernel/Regularizer/mul/x?
)cnn_policy/dense_1/kernel/Regularizer/mulMul4cnn_policy/dense_1/kernel/Regularizer/mul/x:output:02cnn_policy/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/mul?
IdentityIdentity-cnn_policy/dense_1/kernel/Regularizer/mul:z:0<^cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2z
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp
?
?
"__inference_signature_wrapper_2372
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_20512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?:
?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2262	
input
conv2d_14_2224
conv2d_14_2226
conv2d_15_2230
conv2d_15_2232
dense_1_2238
dense_1_2240
identity??=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_14_2224conv2d_14_2226*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_20722#
!conv2d_14/StatefulPartitionedCall?
max_pooling2d/MaxPoolMaxPool*conv2d_14/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????// *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallmax_pooling2d/MaxPool:output:0conv2d_15_2230conv2d_15_2232*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????--@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_21062#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_1/MaxPoolMaxPool*conv2d_15/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? y  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense_1/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0dense_1_2238dense_1_2240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_21422!
dense_1/StatefulPartitionedCall?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_2224*&
_output_shapes
: *
dtype02?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_14/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.cnn_policy/conv2d_14/kernel/Regularizer/Square?
-cnn_policy/conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_14/kernel/Regularizer/Const?
+cnn_policy/conv2d_14/kernel/Regularizer/SumSum2cnn_policy/conv2d_14/kernel/Regularizer/Square:y:06cnn_policy/conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/Sum?
-cnn_policy/conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_14/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_14/kernel/Regularizer/mulMul6cnn_policy/conv2d_14/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/mul?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_2230*&
_output_shapes
: @*
dtype02?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_15/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.cnn_policy/conv2d_15/kernel/Regularizer/Square?
-cnn_policy/conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_15/kernel/Regularizer/Const?
+cnn_policy/conv2d_15/kernel/Regularizer/SumSum2cnn_policy/conv2d_15/kernel/Regularizer/Square:y:06cnn_policy/conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/Sum?
-cnn_policy/conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_15/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_15/kernel/Regularizer/mulMul6cnn_policy/conv2d_15/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/mul?
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2238* 
_output_shapes
:
??*
dtype02=
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
,cnn_policy/dense_1/kernel/Regularizer/SquareSquareCcnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,cnn_policy/dense_1/kernel/Regularizer/Square?
+cnn_policy/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+cnn_policy/dense_1/kernel/Regularizer/Const?
)cnn_policy/dense_1/kernel/Regularizer/SumSum0cnn_policy/dense_1/kernel/Regularizer/Square:y:04cnn_policy/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/Sum?
+cnn_policy/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+cnn_policy/dense_1/kernel/Regularizer/mul/x?
)cnn_policy/dense_1/kernel/Regularizer/mulMul4cnn_policy/dense_1/kernel/Regularizer/mul/x:output:02cnn_policy/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/mul?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0>^cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp>^cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp<^cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::2~
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp2~
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp2z
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?
}
(__inference_conv2d_14_layer_call_fn_2660

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_20722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????^^ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????``::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????``
 
_user_specified_nameinputs
?
?
 __inference__traced_restore_2826
file_prefix0
,assignvariableop_cnn_policy_conv2d_14_kernel0
,assignvariableop_1_cnn_policy_conv2d_14_bias2
.assignvariableop_2_cnn_policy_conv2d_15_kernel0
,assignvariableop_3_cnn_policy_conv2d_15_bias0
,assignvariableop_4_cnn_policy_dense_1_kernel.
*assignvariableop_5_cnn_policy_dense_1_bias

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$f3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp,assignvariableop_cnn_policy_conv2d_14_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp,assignvariableop_1_cnn_policy_conv2d_14_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_cnn_policy_conv2d_15_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp,assignvariableop_3_cnn_policy_conv2d_15_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_cnn_policy_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_cnn_policy_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
__inference__traced_save_2798
file_prefix:
6savev2_cnn_policy_conv2d_14_kernel_read_readvariableop8
4savev2_cnn_policy_conv2d_14_bias_read_readvariableop:
6savev2_cnn_policy_conv2d_15_kernel_read_readvariableop8
4savev2_cnn_policy_conv2d_15_bias_read_readvariableop8
4savev2_cnn_policy_dense_1_kernel_read_readvariableop6
2savev2_cnn_policy_dense_1_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$f3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"f3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_cnn_policy_conv2d_14_kernel_read_readvariableop4savev2_cnn_policy_conv2d_14_bias_read_readvariableop6savev2_cnn_policy_conv2d_15_kernel_read_readvariableop4savev2_cnn_policy_conv2d_15_bias_read_readvariableop4savev2_cnn_policy_dense_1_kernel_read_readvariableop2savev2_cnn_policy_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Y
_input_shapesH
F: : : : @:@:
??:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:&"
 
_output_shapes
:
??: 

_output_shapes
::

_output_shapes
: 
?,
?
__inference__wrapped_model_2051
input_17
3cnn_policy_conv2d_14_conv2d_readvariableop_resource8
4cnn_policy_conv2d_14_biasadd_readvariableop_resource7
3cnn_policy_conv2d_15_conv2d_readvariableop_resource8
4cnn_policy_conv2d_15_biasadd_readvariableop_resource5
1cnn_policy_dense_1_matmul_readvariableop_resource6
2cnn_policy_dense_1_biasadd_readvariableop_resource
identity??+cnn_policy/conv2d_14/BiasAdd/ReadVariableOp?*cnn_policy/conv2d_14/Conv2D/ReadVariableOp?+cnn_policy/conv2d_15/BiasAdd/ReadVariableOp?*cnn_policy/conv2d_15/Conv2D/ReadVariableOp?)cnn_policy/dense_1/BiasAdd/ReadVariableOp?(cnn_policy/dense_1/MatMul/ReadVariableOp?
*cnn_policy/conv2d_14/Conv2D/ReadVariableOpReadVariableOp3cnn_policy_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*cnn_policy/conv2d_14/Conv2D/ReadVariableOp?
cnn_policy/conv2d_14/Conv2DConv2Dinput_12cnn_policy/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ *
paddingVALID*
strides
2
cnn_policy/conv2d_14/Conv2D?
+cnn_policy/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp4cnn_policy_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+cnn_policy/conv2d_14/BiasAdd/ReadVariableOp?
cnn_policy/conv2d_14/BiasAddBiasAdd$cnn_policy/conv2d_14/Conv2D:output:03cnn_policy/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ 2
cnn_policy/conv2d_14/BiasAdd?
cnn_policy/conv2d_14/ReluRelu%cnn_policy/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^^ 2
cnn_policy/conv2d_14/Relu?
 cnn_policy/max_pooling2d/MaxPoolMaxPool'cnn_policy/conv2d_14/Relu:activations:0*/
_output_shapes
:?????????// *
ksize
*
paddingVALID*
strides
2"
 cnn_policy/max_pooling2d/MaxPool?
*cnn_policy/conv2d_15/Conv2D/ReadVariableOpReadVariableOp3cnn_policy_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*cnn_policy/conv2d_15/Conv2D/ReadVariableOp?
cnn_policy/conv2d_15/Conv2DConv2D)cnn_policy/max_pooling2d/MaxPool:output:02cnn_policy/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@*
paddingVALID*
strides
2
cnn_policy/conv2d_15/Conv2D?
+cnn_policy/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp4cnn_policy_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+cnn_policy/conv2d_15/BiasAdd/ReadVariableOp?
cnn_policy/conv2d_15/BiasAddBiasAdd$cnn_policy/conv2d_15/Conv2D:output:03cnn_policy/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@2
cnn_policy/conv2d_15/BiasAdd?
cnn_policy/conv2d_15/ReluRelu%cnn_policy/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????--@2
cnn_policy/conv2d_15/Relu?
"cnn_policy/max_pooling2d_1/MaxPoolMaxPool'cnn_policy/conv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2$
"cnn_policy/max_pooling2d_1/MaxPool?
cnn_policy/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? y  2
cnn_policy/flatten/Const?
cnn_policy/flatten/ReshapeReshape+cnn_policy/max_pooling2d_1/MaxPool:output:0!cnn_policy/flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
cnn_policy/flatten/Reshape?
(cnn_policy/dense_1/MatMul/ReadVariableOpReadVariableOp1cnn_policy_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(cnn_policy/dense_1/MatMul/ReadVariableOp?
cnn_policy/dense_1/MatMulMatMul#cnn_policy/flatten/Reshape:output:00cnn_policy/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cnn_policy/dense_1/MatMul?
)cnn_policy/dense_1/BiasAdd/ReadVariableOpReadVariableOp2cnn_policy_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)cnn_policy/dense_1/BiasAdd/ReadVariableOp?
cnn_policy/dense_1/BiasAddBiasAdd#cnn_policy/dense_1/MatMul:product:01cnn_policy/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
cnn_policy/dense_1/BiasAdd?
cnn_policy/dense_1/SoftmaxSoftmax#cnn_policy/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
cnn_policy/dense_1/Softmax?
IdentityIdentity$cnn_policy/dense_1/Softmax:softmax:0,^cnn_policy/conv2d_14/BiasAdd/ReadVariableOp+^cnn_policy/conv2d_14/Conv2D/ReadVariableOp,^cnn_policy/conv2d_15/BiasAdd/ReadVariableOp+^cnn_policy/conv2d_15/Conv2D/ReadVariableOp*^cnn_policy/dense_1/BiasAdd/ReadVariableOp)^cnn_policy/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::2Z
+cnn_policy/conv2d_14/BiasAdd/ReadVariableOp+cnn_policy/conv2d_14/BiasAdd/ReadVariableOp2X
*cnn_policy/conv2d_14/Conv2D/ReadVariableOp*cnn_policy/conv2d_14/Conv2D/ReadVariableOp2Z
+cnn_policy/conv2d_15/BiasAdd/ReadVariableOp+cnn_policy/conv2d_15/BiasAdd/ReadVariableOp2X
*cnn_policy/conv2d_15/Conv2D/ReadVariableOp*cnn_policy/conv2d_15/Conv2D/ReadVariableOp2V
)cnn_policy/dense_1/BiasAdd/ReadVariableOp)cnn_policy/dense_1/BiasAdd/ReadVariableOp2T
(cnn_policy/dense_1/MatMul/ReadVariableOp(cnn_policy/dense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?
?
)__inference_cnn_policy_layer_call_fn_2483
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_cnn_policy_layer_call_and_return_conditional_losses_22622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?
}
(__inference_conv2d_15_layer_call_fn_2692

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????--@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_21062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????--@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????// ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????// 
 
_user_specified_nameinputs
?H
?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2594	
input,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2Dinput'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ *
paddingVALID*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ 2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^^ 2
conv2d_14/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d_14/Relu:activations:0*/
_output_shapes
:?????????// *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dmax_pooling2d/MaxPool:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@*
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????--@2
conv2d_15/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? y  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_14/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.cnn_policy/conv2d_14/kernel/Regularizer/Square?
-cnn_policy/conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_14/kernel/Regularizer/Const?
+cnn_policy/conv2d_14/kernel/Regularizer/SumSum2cnn_policy/conv2d_14/kernel/Regularizer/Square:y:06cnn_policy/conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/Sum?
-cnn_policy/conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_14/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_14/kernel/Regularizer/mulMul6cnn_policy/conv2d_14/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/mul?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_15/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.cnn_policy/conv2d_15/kernel/Regularizer/Square?
-cnn_policy/conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_15/kernel/Regularizer/Const?
+cnn_policy/conv2d_15/kernel/Regularizer/SumSum2cnn_policy/conv2d_15/kernel/Regularizer/Square:y:06cnn_policy/conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/Sum?
-cnn_policy/conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_15/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_15/kernel/Regularizer/mulMul6cnn_policy/conv2d_15/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/mul?
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
,cnn_policy/dense_1/kernel/Regularizer/SquareSquareCcnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,cnn_policy/dense_1/kernel/Regularizer/Square?
+cnn_policy/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+cnn_policy/dense_1/kernel/Regularizer/Const?
)cnn_policy/dense_1/kernel/Regularizer/SumSum0cnn_policy/dense_1/kernel/Regularizer/Square:y:04cnn_policy/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/Sum?
+cnn_policy/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+cnn_policy/dense_1/kernel/Regularizer/mul/x?
)cnn_policy/dense_1/kernel/Regularizer/mulMul4cnn_policy/dense_1/kernel/Regularizer/mul/x:output:02cnn_policy/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/mul?
IdentityIdentitydense_1/Softmax:softmax:0>^cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp>^cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp<^cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::2~
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp2~
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp2z
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?H
?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2419
input_1,
(conv2d_14_conv2d_readvariableop_resource-
)conv2d_14_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp? conv2d_14/BiasAdd/ReadVariableOp?conv2d_14/Conv2D/ReadVariableOp? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_14/Conv2D/ReadVariableOp?
conv2d_14/Conv2DConv2Dinput_1'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ *
paddingVALID*
strides
2
conv2d_14/Conv2D?
 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_14/BiasAdd/ReadVariableOp?
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ 2
conv2d_14/BiasAdd~
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:?????????^^ 2
conv2d_14/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d_14/Relu:activations:0*/
_output_shapes
:?????????// *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2DConv2Dmax_pooling2d/MaxPool:output:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@*
paddingVALID*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@2
conv2d_15/BiasAdd~
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????--@2
conv2d_15/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_15/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? y  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmax?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_14/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.cnn_policy/conv2d_14/kernel/Regularizer/Square?
-cnn_policy/conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_14/kernel/Regularizer/Const?
+cnn_policy/conv2d_14/kernel/Regularizer/SumSum2cnn_policy/conv2d_14/kernel/Regularizer/Square:y:06cnn_policy/conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/Sum?
-cnn_policy/conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_14/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_14/kernel/Regularizer/mulMul6cnn_policy/conv2d_14/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/mul?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_15/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.cnn_policy/conv2d_15/kernel/Regularizer/Square?
-cnn_policy/conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_15/kernel/Regularizer/Const?
+cnn_policy/conv2d_15/kernel/Regularizer/SumSum2cnn_policy/conv2d_15/kernel/Regularizer/Square:y:06cnn_policy/conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/Sum?
-cnn_policy/conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_15/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_15/kernel/Regularizer/mulMul6cnn_policy/conv2d_15/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/mul?
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
,cnn_policy/dense_1/kernel/Regularizer/SquareSquareCcnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,cnn_policy/dense_1/kernel/Regularizer/Square?
+cnn_policy/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+cnn_policy/dense_1/kernel/Regularizer/Const?
)cnn_policy/dense_1/kernel/Regularizer/SumSum0cnn_policy/dense_1/kernel/Regularizer/Square:y:04cnn_policy/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/Sum?
+cnn_policy/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+cnn_policy/dense_1/kernel/Regularizer/mul/x?
)cnn_policy/dense_1/kernel/Regularizer/mulMul4cnn_policy/dense_1/kernel/Regularizer/mul/x:output:02cnn_policy/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/mul?
IdentityIdentitydense_1/Softmax:softmax:0>^cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp>^cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp<^cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::2~
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp2~
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp2z
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?:
?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2320	
input
conv2d_14_2282
conv2d_14_2284
conv2d_15_2288
conv2d_15_2290
dense_1_2296
dense_1_2298
identity??=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?!conv2d_14/StatefulPartitionedCall?!conv2d_15/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_14_2282conv2d_14_2284*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????^^ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_14_layer_call_and_return_conditional_losses_20722#
!conv2d_14/StatefulPartitionedCall?
max_pooling2d/MaxPoolMaxPool*conv2d_14/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????// *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallmax_pooling2d/MaxPool:output:0conv2d_15_2288conv2d_15_2290*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????--@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_15_layer_call_and_return_conditional_losses_21062#
!conv2d_15/StatefulPartitionedCall?
max_pooling2d_1/MaxPoolMaxPool*conv2d_15/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? y  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape?
dense_1/StatefulPartitionedCallStatefulPartitionedCallflatten/Reshape:output:0dense_1_2296dense_1_2298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_21422!
dense_1/StatefulPartitionedCall?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_2282*&
_output_shapes
: *
dtype02?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_14/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.cnn_policy/conv2d_14/kernel/Regularizer/Square?
-cnn_policy/conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_14/kernel/Regularizer/Const?
+cnn_policy/conv2d_14/kernel/Regularizer/SumSum2cnn_policy/conv2d_14/kernel/Regularizer/Square:y:06cnn_policy/conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/Sum?
-cnn_policy/conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_14/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_14/kernel/Regularizer/mulMul6cnn_policy/conv2d_14/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/mul?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_2288*&
_output_shapes
: @*
dtype02?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_15/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.cnn_policy/conv2d_15/kernel/Regularizer/Square?
-cnn_policy/conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_15/kernel/Regularizer/Const?
+cnn_policy/conv2d_15/kernel/Regularizer/SumSum2cnn_policy/conv2d_15/kernel/Regularizer/Square:y:06cnn_policy/conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/Sum?
-cnn_policy/conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_15/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_15/kernel/Regularizer/mulMul6cnn_policy/conv2d_15/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/mul?
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_2296* 
_output_shapes
:
??*
dtype02=
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp?
,cnn_policy/dense_1/kernel/Regularizer/SquareSquareCcnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,cnn_policy/dense_1/kernel/Regularizer/Square?
+cnn_policy/dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+cnn_policy/dense_1/kernel/Regularizer/Const?
)cnn_policy/dense_1/kernel/Regularizer/SumSum0cnn_policy/dense_1/kernel/Regularizer/Square:y:04cnn_policy/dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/Sum?
+cnn_policy/dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82-
+cnn_policy/dense_1/kernel/Regularizer/mul/x?
)cnn_policy/dense_1/kernel/Regularizer/mulMul4cnn_policy/dense_1/kernel/Regularizer/mul/x:output:02cnn_policy/dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)cnn_policy/dense_1/kernel/Regularizer/mul?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0>^cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp>^cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp<^cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp"^conv2d_14/StatefulPartitionedCall"^conv2d_15/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::2~
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp2~
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp2z
;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp;cnn_policy/dense_1/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?
?
__inference_loss_fn_1_2746J
Fcnn_policy_conv2d_15_kernel_regularizer_square_readvariableop_resource
identity??=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFcnn_policy_conv2d_15_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_15/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.cnn_policy/conv2d_15/kernel/Regularizer/Square?
-cnn_policy/conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_15/kernel/Regularizer/Const?
+cnn_policy/conv2d_15/kernel/Regularizer/SumSum2cnn_policy/conv2d_15/kernel/Regularizer/Square:y:06cnn_policy/conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/Sum?
-cnn_policy/conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_15/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_15/kernel/Regularizer/mulMul6cnn_policy/conv2d_15/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/mul?
IdentityIdentity/cnn_policy/conv2d_15/kernel/Regularizer/mul:z:0>^cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2~
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp
?
?
C__inference_conv2d_15_layer_call_and_return_conditional_losses_2106

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????--@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????--@2
Relu?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02?
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_15/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @20
.cnn_policy/conv2d_15/kernel/Regularizer/Square?
-cnn_policy/conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_15/kernel/Regularizer/Const?
+cnn_policy/conv2d_15/kernel/Regularizer/SumSum2cnn_policy/conv2d_15/kernel/Regularizer/Square:y:06cnn_policy/conv2d_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/Sum?
-cnn_policy/conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_15/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_15/kernel/Regularizer/mulMul6cnn_policy/conv2d_15/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_15/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????--@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????// ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_15/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????// 
 
_user_specified_nameinputs
?
?
)__inference_cnn_policy_layer_call_fn_2500
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_cnn_policy_layer_call_and_return_conditional_losses_23202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?
?
)__inference_cnn_policy_layer_call_fn_2628	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_cnn_policy_layer_call_and_return_conditional_losses_23202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?
?
__inference_loss_fn_0_2735J
Fcnn_policy_conv2d_14_kernel_regularizer_square_readvariableop_resource
identity??=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpFcnn_policy_conv2d_14_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype02?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_14/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.cnn_policy/conv2d_14/kernel/Regularizer/Square?
-cnn_policy/conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_14/kernel/Regularizer/Const?
+cnn_policy/conv2d_14/kernel/Regularizer/SumSum2cnn_policy/conv2d_14/kernel/Regularizer/Square:y:06cnn_policy/conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/Sum?
-cnn_policy/conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_14/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_14/kernel/Regularizer/mulMul6cnn_policy/conv2d_14/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/mul?
IdentityIdentity/cnn_policy/conv2d_14/kernel/Regularizer/mul:z:0>^cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2~
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp
?
?
)__inference_cnn_policy_layer_call_fn_2611	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_cnn_policy_layer_call_and_return_conditional_losses_22622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????``::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?
?
C__inference_conv2d_14_layer_call_and_return_conditional_losses_2072

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????^^ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????^^ 2
Relu?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02?
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp?
.cnn_policy/conv2d_14/kernel/Regularizer/SquareSquareEcnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 20
.cnn_policy/conv2d_14/kernel/Regularizer/Square?
-cnn_policy/conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-cnn_policy/conv2d_14/kernel/Regularizer/Const?
+cnn_policy/conv2d_14/kernel/Regularizer/SumSum2cnn_policy/conv2d_14/kernel/Regularizer/Square:y:06cnn_policy/conv2d_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/Sum?
-cnn_policy/conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82/
-cnn_policy/conv2d_14/kernel/Regularizer/mul/x?
+cnn_policy/conv2d_14/kernel/Regularizer/mulMul6cnn_policy/conv2d_14/kernel/Regularizer/mul/x:output:04cnn_policy/conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2-
+cnn_policy/conv2d_14/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp>^cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????^^ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????``::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2~
=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp=cnn_policy/conv2d_14/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????``
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????``<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?f
?
c1
c2
f3
	variables
trainable_variables
regularization_losses
	keras_api

signatures
/__call__
*0&call_and_return_all_conditional_losses
1_default_save_signature"?
_tf_keras_model?{"class_name": "CnnPolicy", "name": "cnn_policy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "CnnPolicy"}}
?


	kernel

bias
	variables
trainable_variables
regularization_losses
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 96, 96, 4]}}
?


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
4__call__
*5&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 47, 47, 32]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
6__call__
*7&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 12, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30976}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 30976]}}
J
	0

1
2
3
4
5"
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
?
layer_metrics
metrics
	variables
layer_regularization_losses
trainable_variables
regularization_losses

layers
non_trainable_variables
/__call__
1_default_save_signature
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
,
;serving_default"
signature_map
5:3 2cnn_policy/conv2d_14/kernel
':% 2cnn_policy/conv2d_14/bias
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
'
80"
trackable_list_wrapper
?
 non_trainable_variables
!layer_metrics
"metrics
	variables
#layer_regularization_losses
trainable_variables
regularization_losses

$layers
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
5:3 @2cnn_policy/conv2d_15/kernel
':%@2cnn_policy/conv2d_15/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
90"
trackable_list_wrapper
?
%non_trainable_variables
&layer_metrics
'metrics
	variables
(layer_regularization_losses
trainable_variables
regularization_losses

)layers
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
-:+
??2cnn_policy/dense_1/kernel
%:#2cnn_policy/dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
?
*non_trainable_variables
+layer_metrics
,metrics
	variables
-layer_regularization_losses
trainable_variables
regularization_losses

.layers
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
)__inference_cnn_policy_layer_call_fn_2500
)__inference_cnn_policy_layer_call_fn_2628
)__inference_cnn_policy_layer_call_fn_2483
)__inference_cnn_policy_layer_call_fn_2611?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2419
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2547
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2594
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2466?
???
FullArgSpec(
args ?
jself
jinput

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_2051?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????``
?2?
(__inference_conv2d_14_layer_call_fn_2660?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_14_layer_call_and_return_conditional_losses_2651?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_15_layer_call_fn_2692?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_15_layer_call_and_return_conditional_losses_2683?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_2724?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_2715?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_2735?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_2746?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_2757?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
"__inference_signature_wrapper_2372input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_2051w	
8?5
.?+
)?&
input_1?????????``
? "3?0
.
output_1"?
output_1??????????
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2419m	
<?9
2?/
)?&
input_1?????????``
p
? "%?"
?
0?????????
? ?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2466m	
<?9
2?/
)?&
input_1?????????``
p 
? "%?"
?
0?????????
? ?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2547k	
:?7
0?-
'?$
input?????????``
p
? "%?"
?
0?????????
? ?
D__inference_cnn_policy_layer_call_and_return_conditional_losses_2594k	
:?7
0?-
'?$
input?????????``
p 
? "%?"
?
0?????????
? ?
)__inference_cnn_policy_layer_call_fn_2483`	
<?9
2?/
)?&
input_1?????????``
p
? "???????????
)__inference_cnn_policy_layer_call_fn_2500`	
<?9
2?/
)?&
input_1?????????``
p 
? "???????????
)__inference_cnn_policy_layer_call_fn_2611^	
:?7
0?-
'?$
input?????????``
p
? "???????????
)__inference_cnn_policy_layer_call_fn_2628^	
:?7
0?-
'?$
input?????????``
p 
? "???????????
C__inference_conv2d_14_layer_call_and_return_conditional_losses_2651l	
7?4
-?*
(?%
inputs?????????``
? "-?*
#? 
0?????????^^ 
? ?
(__inference_conv2d_14_layer_call_fn_2660_	
7?4
-?*
(?%
inputs?????????``
? " ??????????^^ ?
C__inference_conv2d_15_layer_call_and_return_conditional_losses_2683l7?4
-?*
(?%
inputs?????????// 
? "-?*
#? 
0?????????--@
? ?
(__inference_conv2d_15_layer_call_fn_2692_7?4
-?*
(?%
inputs?????????// 
? " ??????????--@?
A__inference_dense_1_layer_call_and_return_conditional_losses_2715^1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? {
&__inference_dense_1_layer_call_fn_2724Q1?.
'?$
"?
inputs???????????
? "??????????9
__inference_loss_fn_0_2735	?

? 
? "? 9
__inference_loss_fn_1_2746?

? 
? "? 9
__inference_loss_fn_2_2757?

? 
? "? ?
"__inference_signature_wrapper_2372?	
C?@
? 
9?6
4
input_1)?&
input_1?????????``"3?0
.
output_1"?
output_1?????????