??
??
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
$markov_chain_cnn_model/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$markov_chain_cnn_model/conv2d/kernel
?
8markov_chain_cnn_model/conv2d/kernel/Read/ReadVariableOpReadVariableOp$markov_chain_cnn_model/conv2d/kernel*&
_output_shapes
:*
dtype0
?
"markov_chain_cnn_model/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"markov_chain_cnn_model/conv2d/bias
?
6markov_chain_cnn_model/conv2d/bias/Read/ReadVariableOpReadVariableOp"markov_chain_cnn_model/conv2d/bias*
_output_shapes
:*
dtype0
?
&markov_chain_cnn_model/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&markov_chain_cnn_model/conv2d_1/kernel
?
:markov_chain_cnn_model/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp&markov_chain_cnn_model/conv2d_1/kernel*&
_output_shapes
: *
dtype0
?
$markov_chain_cnn_model/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$markov_chain_cnn_model/conv2d_1/bias
?
8markov_chain_cnn_model/conv2d_1/bias/Read/ReadVariableOpReadVariableOp$markov_chain_cnn_model/conv2d_1/bias*
_output_shapes
: *
dtype0
?
&markov_chain_cnn_model/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*7
shared_name(&markov_chain_cnn_model/conv2d_2/kernel
?
:markov_chain_cnn_model/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp&markov_chain_cnn_model/conv2d_2/kernel*&
_output_shapes
: @*
dtype0
?
$markov_chain_cnn_model/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$markov_chain_cnn_model/conv2d_2/bias
?
8markov_chain_cnn_model/conv2d_2/bias/Read/ReadVariableOpReadVariableOp$markov_chain_cnn_model/conv2d_2/bias*
_output_shapes
:@*
dtype0
?
&markov_chain_cnn_model/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`@*7
shared_name(&markov_chain_cnn_model/conv2d_3/kernel
?
:markov_chain_cnn_model/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp&markov_chain_cnn_model/conv2d_3/kernel*&
_output_shapes
:`@*
dtype0
?
$markov_chain_cnn_model/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$markov_chain_cnn_model/conv2d_3/bias
?
8markov_chain_cnn_model/conv2d_3/bias/Read/ReadVariableOpReadVariableOp$markov_chain_cnn_model/conv2d_3/bias*
_output_shapes
:@*
dtype0
?
&markov_chain_cnn_model/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:P *7
shared_name(&markov_chain_cnn_model/conv2d_4/kernel
?
:markov_chain_cnn_model/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp&markov_chain_cnn_model/conv2d_4/kernel*&
_output_shapes
:P *
dtype0
?
$markov_chain_cnn_model/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$markov_chain_cnn_model/conv2d_4/bias
?
8markov_chain_cnn_model/conv2d_4/bias/Read/ReadVariableOpReadVariableOp$markov_chain_cnn_model/conv2d_4/bias*
_output_shapes
: *
dtype0
?
&markov_chain_cnn_model/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&markov_chain_cnn_model/conv2d_5/kernel
?
:markov_chain_cnn_model/conv2d_5/kernel/Read/ReadVariableOpReadVariableOp&markov_chain_cnn_model/conv2d_5/kernel*&
_output_shapes
: *
dtype0
?
$markov_chain_cnn_model/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$markov_chain_cnn_model/conv2d_5/bias
?
8markov_chain_cnn_model/conv2d_5/bias/Read/ReadVariableOpReadVariableOp$markov_chain_cnn_model/conv2d_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
value?!B?! B?!
?
c1
c2
c3
maxpool

upsamp
c1_d
c2_d
c3_d
		variables

trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
V
0
1
2
3
4
5
(6
)7
.8
/9
410
511
V
0
1
2
3
4
5
(6
)7
.8
/9
410
511
 
?
		variables

trainable_variables

:layers
regularization_losses
;layer_regularization_losses
<metrics
=layer_metrics
>non_trainable_variables
 
^\
VARIABLE_VALUE$markov_chain_cnn_model/conv2d/kernel$c1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE"markov_chain_cnn_model/conv2d/bias"c1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

?layers
@layer_regularization_losses
	variables
regularization_losses
trainable_variables
Ametrics
Blayer_metrics
Cnon_trainable_variables
`^
VARIABLE_VALUE&markov_chain_cnn_model/conv2d_1/kernel$c2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE$markov_chain_cnn_model/conv2d_1/bias"c2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Dlayers
Elayer_regularization_losses
	variables
regularization_losses
trainable_variables
Fmetrics
Glayer_metrics
Hnon_trainable_variables
`^
VARIABLE_VALUE&markov_chain_cnn_model/conv2d_2/kernel$c3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE$markov_chain_cnn_model/conv2d_2/bias"c3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Ilayers
Jlayer_regularization_losses
	variables
regularization_losses
trainable_variables
Kmetrics
Llayer_metrics
Mnon_trainable_variables
 
 
 
?

Nlayers
Olayer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
Pmetrics
Qlayer_metrics
Rnon_trainable_variables
 
 
 
?

Slayers
Tlayer_regularization_losses
$	variables
%regularization_losses
&trainable_variables
Umetrics
Vlayer_metrics
Wnon_trainable_variables
b`
VARIABLE_VALUE&markov_chain_cnn_model/conv2d_3/kernel&c1_d/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE$markov_chain_cnn_model/conv2d_3/bias$c1_d/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?

Xlayers
Ylayer_regularization_losses
*	variables
+regularization_losses
,trainable_variables
Zmetrics
[layer_metrics
\non_trainable_variables
b`
VARIABLE_VALUE&markov_chain_cnn_model/conv2d_4/kernel&c2_d/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE$markov_chain_cnn_model/conv2d_4/bias$c2_d/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?

]layers
^layer_regularization_losses
0	variables
1regularization_losses
2trainable_variables
_metrics
`layer_metrics
anon_trainable_variables
b`
VARIABLE_VALUE&markov_chain_cnn_model/conv2d_5/kernel&c3_d/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE$markov_chain_cnn_model/conv2d_5/bias$c3_d/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
?

blayers
clayer_regularization_losses
6	variables
7regularization_losses
8trainable_variables
dmetrics
elayer_metrics
fnon_trainable_variables
8
0
1
2
3
4
5
6
7
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
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1$markov_chain_cnn_model/conv2d/kernel"markov_chain_cnn_model/conv2d/bias&markov_chain_cnn_model/conv2d_1/kernel$markov_chain_cnn_model/conv2d_1/bias&markov_chain_cnn_model/conv2d_2/kernel$markov_chain_cnn_model/conv2d_2/bias&markov_chain_cnn_model/conv2d_3/kernel$markov_chain_cnn_model/conv2d_3/bias&markov_chain_cnn_model/conv2d_4/kernel$markov_chain_cnn_model/conv2d_4/bias&markov_chain_cnn_model/conv2d_5/kernel$markov_chain_cnn_model/conv2d_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_4769
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename8markov_chain_cnn_model/conv2d/kernel/Read/ReadVariableOp6markov_chain_cnn_model/conv2d/bias/Read/ReadVariableOp:markov_chain_cnn_model/conv2d_1/kernel/Read/ReadVariableOp8markov_chain_cnn_model/conv2d_1/bias/Read/ReadVariableOp:markov_chain_cnn_model/conv2d_2/kernel/Read/ReadVariableOp8markov_chain_cnn_model/conv2d_2/bias/Read/ReadVariableOp:markov_chain_cnn_model/conv2d_3/kernel/Read/ReadVariableOp8markov_chain_cnn_model/conv2d_3/bias/Read/ReadVariableOp:markov_chain_cnn_model/conv2d_4/kernel/Read/ReadVariableOp8markov_chain_cnn_model/conv2d_4/bias/Read/ReadVariableOp:markov_chain_cnn_model/conv2d_5/kernel/Read/ReadVariableOp8markov_chain_cnn_model/conv2d_5/bias/Read/ReadVariableOpConst*
Tin
2*
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
__inference__traced_save_5618
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename$markov_chain_cnn_model/conv2d/kernel"markov_chain_cnn_model/conv2d/bias&markov_chain_cnn_model/conv2d_1/kernel$markov_chain_cnn_model/conv2d_1/bias&markov_chain_cnn_model/conv2d_2/kernel$markov_chain_cnn_model/conv2d_2/bias&markov_chain_cnn_model/conv2d_3/kernel$markov_chain_cnn_model/conv2d_3/bias&markov_chain_cnn_model/conv2d_4/kernel$markov_chain_cnn_model/conv2d_4/bias&markov_chain_cnn_model/conv2d_5/kernel$markov_chain_cnn_model/conv2d_5/bias*
Tin
2*
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
 __inference__traced_restore_5664ޥ
?
?
__inference_loss_fn_2_5526U
Qmarkov_chain_cnn_model_conv2d_2_kernel_regularizer_square_readvariableop_resource
identity??Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpQmarkov_chain_cnn_model_conv2d_2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype02J
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2;
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul?
IdentityIdentity:markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul:z:0I^markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_5_5559U
Qmarkov_chain_cnn_model_conv2d_5_kernel_regularizer_square_readvariableop_resource
identity??Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpQmarkov_chain_cnn_model_conv2d_5_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentity:markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul:z:0I^markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4217

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingSAME*
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
:?????????00 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2
Relu?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????00 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????00::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
H
,__inference_max_pooling2d_layer_call_fn_4143

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_41372
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_5139	
input)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????00*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2
conv2d_1/Relu?
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/Reluu
up_sampling2d/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????00@*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2conv2d_1/Relu:activations:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????00`2
concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_3/Reluy
up_sampling2d/Shape_1Shapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape_1?
#up_sampling2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice_1/stack?
%up_sampling2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d/strided_slice_1/stack_1?
%up_sampling2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d/strided_slice_1/stack_2?
up_sampling2d/strided_slice_1StridedSliceup_sampling2d/Shape_1:output:0,up_sampling2d/strided_slice_1/stack:output:0.up_sampling2d/strided_slice_1/stack_1:output:0.up_sampling2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice_1
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mul_1Mul&up_sampling2d/strided_slice_1:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul_1?
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????``@*
half_pixel_centers(2.
,up_sampling2d/resize_1/ResizeNearestNeighbor`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2conv2d/Relu:activations:0=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:?????????``P2

concat_1?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dconcat_1:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` 2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`` 2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
conv2d_5/Relu?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02H
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
7markov_chain_cnn_model/conv2d/kernel/Regularizer/SquareSquareNmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:29
7markov_chain_cnn_model/conv2d/kernel/Regularizer/Square?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             28
6markov_chain_cnn_model/conv2d/kernel/Regularizer/Const?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/SumSum;markov_chain_cnn_model/conv2d/kernel/Regularizer/Square:y:0?markov_chain_cnn_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??828
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mulMul?markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x:output:0=markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02J
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2;
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02J
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`@2;
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02J
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:P 2;
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityconv2d_5/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOpG^markov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpFmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?
?
__inference_loss_fn_3_5537U
Qmarkov_chain_cnn_model_conv2d_3_kernel_regularizer_square_readvariableop_resource
identity??Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpQmarkov_chain_cnn_model_conv2d_3_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:`@*
dtype02J
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`@2;
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul?
IdentityIdentity:markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul:z:0I^markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp
?
z
%__inference_conv2d_layer_call_fn_5333

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
:?????????``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_41832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????``::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????``
 
_user_specified_nameinputs
?
|
'__inference_conv2d_2_layer_call_fn_5397

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
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_42512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
|
'__inference_conv2d_3_layer_call_fn_5429

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
:?????????00@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_42872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????00`::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00`
 
_user_specified_nameinputs
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5420

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
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
:?????????00@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
Relu?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02J
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`@2;
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????00`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????00`
 
_user_specified_nameinputs
?
?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5484

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
Relu?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`` ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????`` 
 
_user_specified_nameinputs
?(
?
__inference__traced_save_5618
file_prefixC
?savev2_markov_chain_cnn_model_conv2d_kernel_read_readvariableopA
=savev2_markov_chain_cnn_model_conv2d_bias_read_readvariableopE
Asavev2_markov_chain_cnn_model_conv2d_1_kernel_read_readvariableopC
?savev2_markov_chain_cnn_model_conv2d_1_bias_read_readvariableopE
Asavev2_markov_chain_cnn_model_conv2d_2_kernel_read_readvariableopC
?savev2_markov_chain_cnn_model_conv2d_2_bias_read_readvariableopE
Asavev2_markov_chain_cnn_model_conv2d_3_kernel_read_readvariableopC
?savev2_markov_chain_cnn_model_conv2d_3_bias_read_readvariableopE
Asavev2_markov_chain_cnn_model_conv2d_4_kernel_read_readvariableopC
?savev2_markov_chain_cnn_model_conv2d_4_bias_read_readvariableopE
Asavev2_markov_chain_cnn_model_conv2d_5_kernel_read_readvariableopC
?savev2_markov_chain_cnn_model_conv2d_5_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$c3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c3/bias/.ATTRIBUTES/VARIABLE_VALUEB&c1_d/kernel/.ATTRIBUTES/VARIABLE_VALUEB$c1_d/bias/.ATTRIBUTES/VARIABLE_VALUEB&c2_d/kernel/.ATTRIBUTES/VARIABLE_VALUEB$c2_d/bias/.ATTRIBUTES/VARIABLE_VALUEB&c3_d/kernel/.ATTRIBUTES/VARIABLE_VALUEB$c3_d/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0?savev2_markov_chain_cnn_model_conv2d_kernel_read_readvariableop=savev2_markov_chain_cnn_model_conv2d_bias_read_readvariableopAsavev2_markov_chain_cnn_model_conv2d_1_kernel_read_readvariableop?savev2_markov_chain_cnn_model_conv2d_1_bias_read_readvariableopAsavev2_markov_chain_cnn_model_conv2d_2_kernel_read_readvariableop?savev2_markov_chain_cnn_model_conv2d_2_bias_read_readvariableopAsavev2_markov_chain_cnn_model_conv2d_3_kernel_read_readvariableop?savev2_markov_chain_cnn_model_conv2d_3_bias_read_readvariableopAsavev2_markov_chain_cnn_model_conv2d_4_kernel_read_readvariableop?savev2_markov_chain_cnn_model_conv2d_4_bias_read_readvariableopAsavev2_markov_chain_cnn_model_conv2d_5_kernel_read_readvariableop?savev2_markov_chain_cnn_model_conv2d_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : @:@:`@:@:P : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:`@: 

_output_shapes
:@:,	(
&
_output_shapes
:P : 


_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_5324

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
Relu?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02H
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
7markov_chain_cnn_model/conv2d/kernel/Regularizer/SquareSquareNmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:29
7markov_chain_cnn_model/conv2d/kernel/Regularizer/Square?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             28
6markov_chain_cnn_model/conv2d/kernel/Regularizer/Const?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/SumSum;markov_chain_cnn_model/conv2d/kernel/Regularizer/Square:y:0?markov_chain_cnn_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??828
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mulMul?markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x:output:0=markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpG^markov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????``::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpFmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????``
 
_user_specified_nameinputs
?
?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5452

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` *
paddingSAME*
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
:?????????`` 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????`` 2
Relu?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02J
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:P 2;
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????`` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????``P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????``P
 
_user_specified_nameinputs
˃
?
__inference__wrapped_model_4131
input_1@
<markov_chain_cnn_model_conv2d_conv2d_readvariableop_resourceA
=markov_chain_cnn_model_conv2d_biasadd_readvariableop_resourceB
>markov_chain_cnn_model_conv2d_1_conv2d_readvariableop_resourceC
?markov_chain_cnn_model_conv2d_1_biasadd_readvariableop_resourceB
>markov_chain_cnn_model_conv2d_2_conv2d_readvariableop_resourceC
?markov_chain_cnn_model_conv2d_2_biasadd_readvariableop_resourceB
>markov_chain_cnn_model_conv2d_3_conv2d_readvariableop_resourceC
?markov_chain_cnn_model_conv2d_3_biasadd_readvariableop_resourceB
>markov_chain_cnn_model_conv2d_4_conv2d_readvariableop_resourceC
?markov_chain_cnn_model_conv2d_4_biasadd_readvariableop_resourceB
>markov_chain_cnn_model_conv2d_5_conv2d_readvariableop_resourceC
?markov_chain_cnn_model_conv2d_5_biasadd_readvariableop_resource
identity??4markov_chain_cnn_model/conv2d/BiasAdd/ReadVariableOp?3markov_chain_cnn_model/conv2d/Conv2D/ReadVariableOp?6markov_chain_cnn_model/conv2d_1/BiasAdd/ReadVariableOp?5markov_chain_cnn_model/conv2d_1/Conv2D/ReadVariableOp?6markov_chain_cnn_model/conv2d_2/BiasAdd/ReadVariableOp?5markov_chain_cnn_model/conv2d_2/Conv2D/ReadVariableOp?6markov_chain_cnn_model/conv2d_3/BiasAdd/ReadVariableOp?5markov_chain_cnn_model/conv2d_3/Conv2D/ReadVariableOp?6markov_chain_cnn_model/conv2d_4/BiasAdd/ReadVariableOp?5markov_chain_cnn_model/conv2d_4/Conv2D/ReadVariableOp?6markov_chain_cnn_model/conv2d_5/BiasAdd/ReadVariableOp?5markov_chain_cnn_model/conv2d_5/Conv2D/ReadVariableOp?
3markov_chain_cnn_model/conv2d/Conv2D/ReadVariableOpReadVariableOp<markov_chain_cnn_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3markov_chain_cnn_model/conv2d/Conv2D/ReadVariableOp?
$markov_chain_cnn_model/conv2d/Conv2DConv2Dinput_1;markov_chain_cnn_model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2&
$markov_chain_cnn_model/conv2d/Conv2D?
4markov_chain_cnn_model/conv2d/BiasAdd/ReadVariableOpReadVariableOp=markov_chain_cnn_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4markov_chain_cnn_model/conv2d/BiasAdd/ReadVariableOp?
%markov_chain_cnn_model/conv2d/BiasAddBiasAdd-markov_chain_cnn_model/conv2d/Conv2D:output:0<markov_chain_cnn_model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2'
%markov_chain_cnn_model/conv2d/BiasAdd?
"markov_chain_cnn_model/conv2d/ReluRelu.markov_chain_cnn_model/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2$
"markov_chain_cnn_model/conv2d/Relu?
,markov_chain_cnn_model/max_pooling2d/MaxPoolMaxPool0markov_chain_cnn_model/conv2d/Relu:activations:0*/
_output_shapes
:?????????00*
ksize
*
paddingVALID*
strides
2.
,markov_chain_cnn_model/max_pooling2d/MaxPool?
5markov_chain_cnn_model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp>markov_chain_cnn_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5markov_chain_cnn_model/conv2d_1/Conv2D/ReadVariableOp?
&markov_chain_cnn_model/conv2d_1/Conv2DConv2D5markov_chain_cnn_model/max_pooling2d/MaxPool:output:0=markov_chain_cnn_model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingSAME*
strides
2(
&markov_chain_cnn_model/conv2d_1/Conv2D?
6markov_chain_cnn_model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?markov_chain_cnn_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6markov_chain_cnn_model/conv2d_1/BiasAdd/ReadVariableOp?
'markov_chain_cnn_model/conv2d_1/BiasAddBiasAdd/markov_chain_cnn_model/conv2d_1/Conv2D:output:0>markov_chain_cnn_model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 2)
'markov_chain_cnn_model/conv2d_1/BiasAdd?
$markov_chain_cnn_model/conv2d_1/ReluRelu0markov_chain_cnn_model/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2&
$markov_chain_cnn_model/conv2d_1/Relu?
.markov_chain_cnn_model/max_pooling2d/MaxPool_1MaxPool2markov_chain_cnn_model/conv2d_1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
20
.markov_chain_cnn_model/max_pooling2d/MaxPool_1?
5markov_chain_cnn_model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp>markov_chain_cnn_model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype027
5markov_chain_cnn_model/conv2d_2/Conv2D/ReadVariableOp?
&markov_chain_cnn_model/conv2d_2/Conv2DConv2D7markov_chain_cnn_model/max_pooling2d/MaxPool_1:output:0=markov_chain_cnn_model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2(
&markov_chain_cnn_model/conv2d_2/Conv2D?
6markov_chain_cnn_model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?markov_chain_cnn_model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6markov_chain_cnn_model/conv2d_2/BiasAdd/ReadVariableOp?
'markov_chain_cnn_model/conv2d_2/BiasAddBiasAdd/markov_chain_cnn_model/conv2d_2/Conv2D:output:0>markov_chain_cnn_model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2)
'markov_chain_cnn_model/conv2d_2/BiasAdd?
$markov_chain_cnn_model/conv2d_2/ReluRelu0markov_chain_cnn_model/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2&
$markov_chain_cnn_model/conv2d_2/Relu?
*markov_chain_cnn_model/up_sampling2d/ShapeShape2markov_chain_cnn_model/conv2d_2/Relu:activations:0*
T0*
_output_shapes
:2,
*markov_chain_cnn_model/up_sampling2d/Shape?
8markov_chain_cnn_model/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8markov_chain_cnn_model/up_sampling2d/strided_slice/stack?
:markov_chain_cnn_model/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:markov_chain_cnn_model/up_sampling2d/strided_slice/stack_1?
:markov_chain_cnn_model/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:markov_chain_cnn_model/up_sampling2d/strided_slice/stack_2?
2markov_chain_cnn_model/up_sampling2d/strided_sliceStridedSlice3markov_chain_cnn_model/up_sampling2d/Shape:output:0Amarkov_chain_cnn_model/up_sampling2d/strided_slice/stack:output:0Cmarkov_chain_cnn_model/up_sampling2d/strided_slice/stack_1:output:0Cmarkov_chain_cnn_model/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:24
2markov_chain_cnn_model/up_sampling2d/strided_slice?
*markov_chain_cnn_model/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2,
*markov_chain_cnn_model/up_sampling2d/Const?
(markov_chain_cnn_model/up_sampling2d/mulMul;markov_chain_cnn_model/up_sampling2d/strided_slice:output:03markov_chain_cnn_model/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2*
(markov_chain_cnn_model/up_sampling2d/mul?
Amarkov_chain_cnn_model/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor2markov_chain_cnn_model/conv2d_2/Relu:activations:0,markov_chain_cnn_model/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????00@*
half_pixel_centers(2C
Amarkov_chain_cnn_model/up_sampling2d/resize/ResizeNearestNeighbor?
"markov_chain_cnn_model/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"markov_chain_cnn_model/concat/axis?
markov_chain_cnn_model/concatConcatV22markov_chain_cnn_model/conv2d_1/Relu:activations:0Rmarkov_chain_cnn_model/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0+markov_chain_cnn_model/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????00`2
markov_chain_cnn_model/concat?
5markov_chain_cnn_model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp>markov_chain_cnn_model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype027
5markov_chain_cnn_model/conv2d_3/Conv2D/ReadVariableOp?
&markov_chain_cnn_model/conv2d_3/Conv2DConv2D&markov_chain_cnn_model/concat:output:0=markov_chain_cnn_model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2(
&markov_chain_cnn_model/conv2d_3/Conv2D?
6markov_chain_cnn_model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?markov_chain_cnn_model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6markov_chain_cnn_model/conv2d_3/BiasAdd/ReadVariableOp?
'markov_chain_cnn_model/conv2d_3/BiasAddBiasAdd/markov_chain_cnn_model/conv2d_3/Conv2D:output:0>markov_chain_cnn_model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2)
'markov_chain_cnn_model/conv2d_3/BiasAdd?
$markov_chain_cnn_model/conv2d_3/ReluRelu0markov_chain_cnn_model/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2&
$markov_chain_cnn_model/conv2d_3/Relu?
,markov_chain_cnn_model/up_sampling2d/Shape_1Shape2markov_chain_cnn_model/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2.
,markov_chain_cnn_model/up_sampling2d/Shape_1?
:markov_chain_cnn_model/up_sampling2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:markov_chain_cnn_model/up_sampling2d/strided_slice_1/stack?
<markov_chain_cnn_model/up_sampling2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<markov_chain_cnn_model/up_sampling2d/strided_slice_1/stack_1?
<markov_chain_cnn_model/up_sampling2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<markov_chain_cnn_model/up_sampling2d/strided_slice_1/stack_2?
4markov_chain_cnn_model/up_sampling2d/strided_slice_1StridedSlice5markov_chain_cnn_model/up_sampling2d/Shape_1:output:0Cmarkov_chain_cnn_model/up_sampling2d/strided_slice_1/stack:output:0Emarkov_chain_cnn_model/up_sampling2d/strided_slice_1/stack_1:output:0Emarkov_chain_cnn_model/up_sampling2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:26
4markov_chain_cnn_model/up_sampling2d/strided_slice_1?
,markov_chain_cnn_model/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2.
,markov_chain_cnn_model/up_sampling2d/Const_1?
*markov_chain_cnn_model/up_sampling2d/mul_1Mul=markov_chain_cnn_model/up_sampling2d/strided_slice_1:output:05markov_chain_cnn_model/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2,
*markov_chain_cnn_model/up_sampling2d/mul_1?
Cmarkov_chain_cnn_model/up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighbor2markov_chain_cnn_model/conv2d_3/Relu:activations:0.markov_chain_cnn_model/up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????``@*
half_pixel_centers(2E
Cmarkov_chain_cnn_model/up_sampling2d/resize_1/ResizeNearestNeighbor?
$markov_chain_cnn_model/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$markov_chain_cnn_model/concat_1/axis?
markov_chain_cnn_model/concat_1ConcatV20markov_chain_cnn_model/conv2d/Relu:activations:0Tmarkov_chain_cnn_model/up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0-markov_chain_cnn_model/concat_1/axis:output:0*
N*
T0*/
_output_shapes
:?????????``P2!
markov_chain_cnn_model/concat_1?
5markov_chain_cnn_model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp>markov_chain_cnn_model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:P *
dtype027
5markov_chain_cnn_model/conv2d_4/Conv2D/ReadVariableOp?
&markov_chain_cnn_model/conv2d_4/Conv2DConv2D(markov_chain_cnn_model/concat_1:output:0=markov_chain_cnn_model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` *
paddingSAME*
strides
2(
&markov_chain_cnn_model/conv2d_4/Conv2D?
6markov_chain_cnn_model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?markov_chain_cnn_model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6markov_chain_cnn_model/conv2d_4/BiasAdd/ReadVariableOp?
'markov_chain_cnn_model/conv2d_4/BiasAddBiasAdd/markov_chain_cnn_model/conv2d_4/Conv2D:output:0>markov_chain_cnn_model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` 2)
'markov_chain_cnn_model/conv2d_4/BiasAdd?
$markov_chain_cnn_model/conv2d_4/ReluRelu0markov_chain_cnn_model/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`` 2&
$markov_chain_cnn_model/conv2d_4/Relu?
5markov_chain_cnn_model/conv2d_5/Conv2D/ReadVariableOpReadVariableOp>markov_chain_cnn_model_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5markov_chain_cnn_model/conv2d_5/Conv2D/ReadVariableOp?
&markov_chain_cnn_model/conv2d_5/Conv2DConv2D2markov_chain_cnn_model/conv2d_4/Relu:activations:0=markov_chain_cnn_model/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2(
&markov_chain_cnn_model/conv2d_5/Conv2D?
6markov_chain_cnn_model/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?markov_chain_cnn_model_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6markov_chain_cnn_model/conv2d_5/BiasAdd/ReadVariableOp?
'markov_chain_cnn_model/conv2d_5/BiasAddBiasAdd/markov_chain_cnn_model/conv2d_5/Conv2D:output:0>markov_chain_cnn_model/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2)
'markov_chain_cnn_model/conv2d_5/BiasAdd?
$markov_chain_cnn_model/conv2d_5/ReluRelu0markov_chain_cnn_model/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2&
$markov_chain_cnn_model/conv2d_5/Relu?
IdentityIdentity2markov_chain_cnn_model/conv2d_5/Relu:activations:05^markov_chain_cnn_model/conv2d/BiasAdd/ReadVariableOp4^markov_chain_cnn_model/conv2d/Conv2D/ReadVariableOp7^markov_chain_cnn_model/conv2d_1/BiasAdd/ReadVariableOp6^markov_chain_cnn_model/conv2d_1/Conv2D/ReadVariableOp7^markov_chain_cnn_model/conv2d_2/BiasAdd/ReadVariableOp6^markov_chain_cnn_model/conv2d_2/Conv2D/ReadVariableOp7^markov_chain_cnn_model/conv2d_3/BiasAdd/ReadVariableOp6^markov_chain_cnn_model/conv2d_3/Conv2D/ReadVariableOp7^markov_chain_cnn_model/conv2d_4/BiasAdd/ReadVariableOp6^markov_chain_cnn_model/conv2d_4/Conv2D/ReadVariableOp7^markov_chain_cnn_model/conv2d_5/BiasAdd/ReadVariableOp6^markov_chain_cnn_model/conv2d_5/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::2l
4markov_chain_cnn_model/conv2d/BiasAdd/ReadVariableOp4markov_chain_cnn_model/conv2d/BiasAdd/ReadVariableOp2j
3markov_chain_cnn_model/conv2d/Conv2D/ReadVariableOp3markov_chain_cnn_model/conv2d/Conv2D/ReadVariableOp2p
6markov_chain_cnn_model/conv2d_1/BiasAdd/ReadVariableOp6markov_chain_cnn_model/conv2d_1/BiasAdd/ReadVariableOp2n
5markov_chain_cnn_model/conv2d_1/Conv2D/ReadVariableOp5markov_chain_cnn_model/conv2d_1/Conv2D/ReadVariableOp2p
6markov_chain_cnn_model/conv2d_2/BiasAdd/ReadVariableOp6markov_chain_cnn_model/conv2d_2/BiasAdd/ReadVariableOp2n
5markov_chain_cnn_model/conv2d_2/Conv2D/ReadVariableOp5markov_chain_cnn_model/conv2d_2/Conv2D/ReadVariableOp2p
6markov_chain_cnn_model/conv2d_3/BiasAdd/ReadVariableOp6markov_chain_cnn_model/conv2d_3/BiasAdd/ReadVariableOp2n
5markov_chain_cnn_model/conv2d_3/Conv2D/ReadVariableOp5markov_chain_cnn_model/conv2d_3/Conv2D/ReadVariableOp2p
6markov_chain_cnn_model/conv2d_4/BiasAdd/ReadVariableOp6markov_chain_cnn_model/conv2d_4/BiasAdd/ReadVariableOp2n
5markov_chain_cnn_model/conv2d_4/Conv2D/ReadVariableOp5markov_chain_cnn_model/conv2d_4/Conv2D/ReadVariableOp2p
6markov_chain_cnn_model/conv2d_5/BiasAdd/ReadVariableOp6markov_chain_cnn_model/conv2d_5/BiasAdd/ReadVariableOp2n
5markov_chain_cnn_model/conv2d_5/Conv2D/ReadVariableOp5markov_chain_cnn_model/conv2d_5/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5388

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
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
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02J
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2;
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5356

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingSAME*
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
:?????????00 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2
Relu?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????00 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????00::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4251

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
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
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02J
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2;
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
H
,__inference_up_sampling2d_layer_call_fn_4162

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_41562
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?7
?
 __inference__traced_restore_5664
file_prefix9
5assignvariableop_markov_chain_cnn_model_conv2d_kernel9
5assignvariableop_1_markov_chain_cnn_model_conv2d_bias=
9assignvariableop_2_markov_chain_cnn_model_conv2d_1_kernel;
7assignvariableop_3_markov_chain_cnn_model_conv2d_1_bias=
9assignvariableop_4_markov_chain_cnn_model_conv2d_2_kernel;
7assignvariableop_5_markov_chain_cnn_model_conv2d_2_bias=
9assignvariableop_6_markov_chain_cnn_model_conv2d_3_kernel;
7assignvariableop_7_markov_chain_cnn_model_conv2d_3_bias=
9assignvariableop_8_markov_chain_cnn_model_conv2d_4_kernel;
7assignvariableop_9_markov_chain_cnn_model_conv2d_4_bias>
:assignvariableop_10_markov_chain_cnn_model_conv2d_5_kernel<
8assignvariableop_11_markov_chain_cnn_model_conv2d_5_bias
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B$c1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c1/bias/.ATTRIBUTES/VARIABLE_VALUEB$c2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c2/bias/.ATTRIBUTES/VARIABLE_VALUEB$c3/kernel/.ATTRIBUTES/VARIABLE_VALUEB"c3/bias/.ATTRIBUTES/VARIABLE_VALUEB&c1_d/kernel/.ATTRIBUTES/VARIABLE_VALUEB$c1_d/bias/.ATTRIBUTES/VARIABLE_VALUEB&c2_d/kernel/.ATTRIBUTES/VARIABLE_VALUEB$c2_d/bias/.ATTRIBUTES/VARIABLE_VALUEB&c3_d/kernel/.ATTRIBUTES/VARIABLE_VALUEB$c3_d/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp5assignvariableop_markov_chain_cnn_model_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp5assignvariableop_1_markov_chain_cnn_model_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp9assignvariableop_2_markov_chain_cnn_model_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp7assignvariableop_3_markov_chain_cnn_model_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp9assignvariableop_4_markov_chain_cnn_model_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp7assignvariableop_5_markov_chain_cnn_model_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp9assignvariableop_6_markov_chain_cnn_model_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp7assignvariableop_7_markov_chain_cnn_model_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp9assignvariableop_8_markov_chain_cnn_model_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp7assignvariableop_9_markov_chain_cnn_model_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_markov_chain_cnn_model_conv2d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp8assignvariableop_11_markov_chain_cnn_model_conv2d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
__inference_loss_fn_1_5515U
Qmarkov_chain_cnn_model_conv2d_1_kernel_regularizer_square_readvariableop_resource
identity??Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpQmarkov_chain_cnn_model_conv2d_1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul?
IdentityIdentity:markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul:z:0I^markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp
?
?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4323

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` *
paddingSAME*
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
:?????????`` 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????`` 2
Relu?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02J
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:P 2;
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????`` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????``P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????``P
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_5504S
Omarkov_chain_cnn_model_conv2d_kernel_regularizer_square_readvariableop_resource
identity??Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpOmarkov_chain_cnn_model_conv2d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:*
dtype02H
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
7markov_chain_cnn_model/conv2d/kernel/Regularizer/SquareSquareNmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:29
7markov_chain_cnn_model/conv2d/kernel/Regularizer/Square?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             28
6markov_chain_cnn_model/conv2d/kernel/Regularizer/Const?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/SumSum;markov_chain_cnn_model/conv2d/kernel/Regularizer/Square:y:0?markov_chain_cnn_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??828
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mulMul?markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x:output:0=markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mul?
IdentityIdentity8markov_chain_cnn_model/conv2d/kernel/Regularizer/mul:z:0G^markov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpFmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp
?	
?
5__inference_markov_chain_cnn_model_layer_call_fn_5272	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_45682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
??
?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_4977
input_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????00*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2
conv2d_1/Relu?
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/Reluu
up_sampling2d/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????00@*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2conv2d_1/Relu:activations:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????00`2
concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_3/Reluy
up_sampling2d/Shape_1Shapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape_1?
#up_sampling2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice_1/stack?
%up_sampling2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d/strided_slice_1/stack_1?
%up_sampling2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d/strided_slice_1/stack_2?
up_sampling2d/strided_slice_1StridedSliceup_sampling2d/Shape_1:output:0,up_sampling2d/strided_slice_1/stack:output:0.up_sampling2d/strided_slice_1/stack_1:output:0.up_sampling2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice_1
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mul_1Mul&up_sampling2d/strided_slice_1:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul_1?
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????``@*
half_pixel_centers(2.
,up_sampling2d/resize_1/ResizeNearestNeighbor`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2conv2d/Relu:activations:0=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:?????????``P2

concat_1?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dconcat_1:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` 2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`` 2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
conv2d_5/Relu?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02H
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
7markov_chain_cnn_model/conv2d/kernel/Regularizer/SquareSquareNmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:29
7markov_chain_cnn_model/conv2d/kernel/Regularizer/Square?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             28
6markov_chain_cnn_model/conv2d/kernel/Regularizer/Const?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/SumSum;markov_chain_cnn_model/conv2d/kernel/Regularizer/Square:y:0?markov_chain_cnn_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??828
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mulMul?markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x:output:0=markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02J
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2;
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02J
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`@2;
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02J
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:P 2;
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityconv2d_5/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOpG^markov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpFmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
??
?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_4568	
input
conv2d_4493
conv2d_4495
conv2d_1_4499
conv2d_1_4501
conv2d_2_4505
conv2d_2_4507
conv2d_3_4513
conv2d_3_4515
conv2d_4_4521
conv2d_4_4523
conv2d_5_4526
conv2d_5_4528
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_4493conv2d_4495*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_41832 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_41372
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_4499conv2d_1_4501*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_42172"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d/PartitionedCall_1PartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_41372!
max_pooling2d/PartitionedCall_1?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d/PartitionedCall_1:output:0conv2d_2_4505conv2d_2_4507*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_42512"
 conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_41562
up_sampling2d/PartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2)conv2d_1/StatefulPartitionedCall:output:0&up_sampling2d/PartitionedCall:output:0concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????00`2
concat?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0conv2d_3_4513conv2d_3_4515*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_42872"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d/PartitionedCall_1PartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_41562!
up_sampling2d/PartitionedCall_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2'conv2d/StatefulPartitionedCall:output:0(up_sampling2d/PartitionedCall_1:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:?????????``P2

concat_1?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0conv2d_4_4521conv2d_4_4523*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`` *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_43232"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_4526conv2d_5_4528*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_43562"
 conv2d_5/StatefulPartitionedCall?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4493*&
_output_shapes
:*
dtype02H
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
7markov_chain_cnn_model/conv2d/kernel/Regularizer/SquareSquareNmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:29
7markov_chain_cnn_model/conv2d/kernel/Regularizer/Square?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             28
6markov_chain_cnn_model/conv2d/kernel/Regularizer/Const?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/SumSum;markov_chain_cnn_model/conv2d/kernel/Regularizer/Square:y:0?markov_chain_cnn_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??828
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mulMul?markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x:output:0=markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_4499*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_4505*&
_output_shapes
: @*
dtype02J
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2;
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_4513*&
_output_shapes
:`@*
dtype02J
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`@2;
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_4521*&
_output_shapes
:P *
dtype02J
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:P 2;
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_4526*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentity)conv2d_5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCallG^markov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpFmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?
|
'__inference_conv2d_4_layer_call_fn_5461

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
:?????????`` *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_43232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????`` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????``P::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????``P
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_5548U
Qmarkov_chain_cnn_model_conv2d_4_kernel_regularizer_square_readvariableop_resource
identity??Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpQmarkov_chain_cnn_model_conv2d_4_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:P *
dtype02J
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:P 2;
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul?
IdentityIdentity:markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul:z:0I^markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp
?	
?
5__inference_markov_chain_cnn_model_layer_call_fn_5301	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_46752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?
?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4287

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
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
:?????????00@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
Relu?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02J
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`@2;
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????00@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????00`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????00`
 
_user_specified_nameinputs
??
?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_4675	
input
conv2d_4600
conv2d_4602
conv2d_1_4606
conv2d_1_4608
conv2d_2_4612
conv2d_2_4614
conv2d_3_4620
conv2d_3_4622
conv2d_4_4628
conv2d_4_4630
conv2d_5_4633
conv2d_5_4635
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputconv2d_4600conv2d_4602*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_41832 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_41372
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_4606conv2d_1_4608*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_42172"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d/PartitionedCall_1PartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_41372!
max_pooling2d/PartitionedCall_1?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d/PartitionedCall_1:output:0conv2d_2_4612conv2d_2_4614*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_42512"
 conv2d_2/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_41562
up_sampling2d/PartitionedCall\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2)conv2d_1/StatefulPartitionedCall:output:0&up_sampling2d/PartitionedCall:output:0concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????00`2
concat?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconcat:output:0conv2d_3_4620conv2d_3_4622*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????00@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_42872"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d/PartitionedCall_1PartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_41562!
up_sampling2d/PartitionedCall_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2'conv2d/StatefulPartitionedCall:output:0(up_sampling2d/PartitionedCall_1:output:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:?????????``P2

concat_1?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0conv2d_4_4628conv2d_4_4630*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`` *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_43232"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_4633conv2d_5_4635*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_43562"
 conv2d_5/StatefulPartitionedCall?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4600*&
_output_shapes
:*
dtype02H
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
7markov_chain_cnn_model/conv2d/kernel/Regularizer/SquareSquareNmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:29
7markov_chain_cnn_model/conv2d/kernel/Regularizer/Square?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             28
6markov_chain_cnn_model/conv2d/kernel/Regularizer/Const?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/SumSum;markov_chain_cnn_model/conv2d/kernel/Regularizer/Square:y:0?markov_chain_cnn_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??828
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mulMul?markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x:output:0=markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_4606*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_4612*&
_output_shapes
: @*
dtype02J
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2;
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_4620*&
_output_shapes
:`@*
dtype02J
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`@2;
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_4628*&
_output_shapes
:P *
dtype02J
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:P 2;
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_4633*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentity)conv2d_5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCallG^markov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpFmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
??
?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_5243	
input)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????00*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2
conv2d_1/Relu?
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/Reluu
up_sampling2d/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????00@*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2conv2d_1/Relu:activations:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????00`2
concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_3/Reluy
up_sampling2d/Shape_1Shapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape_1?
#up_sampling2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice_1/stack?
%up_sampling2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d/strided_slice_1/stack_1?
%up_sampling2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d/strided_slice_1/stack_2?
up_sampling2d/strided_slice_1StridedSliceup_sampling2d/Shape_1:output:0,up_sampling2d/strided_slice_1/stack:output:0.up_sampling2d/strided_slice_1/stack_1:output:0.up_sampling2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice_1
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mul_1Mul&up_sampling2d/strided_slice_1:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul_1?
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????``@*
half_pixel_centers(2.
,up_sampling2d/resize_1/ResizeNearestNeighbor`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2conv2d/Relu:activations:0=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:?????????``P2

concat_1?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dconcat_1:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` 2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`` 2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
conv2d_5/Relu?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02H
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
7markov_chain_cnn_model/conv2d/kernel/Regularizer/SquareSquareNmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:29
7markov_chain_cnn_model/conv2d/kernel/Regularizer/Square?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             28
6markov_chain_cnn_model/conv2d/kernel/Regularizer/Const?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/SumSum;markov_chain_cnn_model/conv2d/kernel/Regularizer/Square:y:0?markov_chain_cnn_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??828
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mulMul?markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x:output:0=markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02J
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2;
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02J
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`@2;
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02J
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:P 2;
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityconv2d_5/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOpG^markov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpFmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:V R
/
_output_shapes
:?????????``

_user_specified_nameinput
?
|
'__inference_conv2d_5_layer_call_fn_5493

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
:?????????``*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_43562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`` ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`` 
 
_user_specified_nameinputs
?	
?
"__inference_signature_wrapper_4769
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_41312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4137

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4156

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_4873
input_1)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinput_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????00*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00 2
conv2d_1/Relu?
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool_1?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/Reluu
up_sampling2d/ShapeShapeconv2d_2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_2/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????00@*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2conv2d_1/Relu:activations:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????00`2
concat?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconcat:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????00@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????00@2
conv2d_3/Reluy
up_sampling2d/Shape_1Shapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape_1?
#up_sampling2d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice_1/stack?
%up_sampling2d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d/strided_slice_1/stack_1?
%up_sampling2d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d/strided_slice_1/stack_2?
up_sampling2d/strided_slice_1StridedSliceup_sampling2d/Shape_1:output:0,up_sampling2d/strided_slice_1/stack:output:0.up_sampling2d/strided_slice_1/stack_1:output:0.up_sampling2d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice_1
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mul_1Mul&up_sampling2d/strided_slice_1:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul_1?
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????``@*
half_pixel_centers(2.
,up_sampling2d/resize_1/ResizeNearestNeighbor`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis?
concat_1ConcatV2conv2d/Relu:activations:0=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0concat_1/axis:output:0*
N*
T0*/
_output_shapes
:?????????``P2

concat_1?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dconcat_1:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`` 2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????`` 2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
conv2d_5/Relu?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02H
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
7markov_chain_cnn_model/conv2d/kernel/Regularizer/SquareSquareNmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:29
7markov_chain_cnn_model/conv2d/kernel/Regularizer/Square?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             28
6markov_chain_cnn_model/conv2d/kernel/Regularizer/Const?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/SumSum;markov_chain_cnn_model/conv2d/kernel/Regularizer/Square:y:0?markov_chain_cnn_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??828
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mulMul?markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x:output:0=markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_1/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02J
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2;
9markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_2/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:`@*
dtype02J
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:`@2;
9markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_3/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:P *
dtype02J
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:P 2;
9markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_4/kernel/Regularizer/mul?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityconv2d_5/Relu:activations:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOpG^markov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpI^markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpFmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_1/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_2/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_3/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_4/kernel/Regularizer/Square/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?	
?
5__inference_markov_chain_cnn_model_layer_call_fn_5035
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_46752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1
?
|
'__inference_conv2d_1_layer_call_fn_5365

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
:?????????00 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_42172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????00 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????00::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????00
 
_user_specified_nameinputs
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_4183

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
Relu?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02H
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp?
7markov_chain_cnn_model/conv2d/kernel/Regularizer/SquareSquareNmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:29
7markov_chain_cnn_model/conv2d/kernel/Regularizer/Square?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             28
6markov_chain_cnn_model/conv2d/kernel/Regularizer/Const?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/SumSum;markov_chain_cnn_model/conv2d/kernel/Regularizer/Square:y:0?markov_chain_cnn_model/conv2d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum?
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??828
6markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x?
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mulMul?markov_chain_cnn_model/conv2d/kernel/Regularizer/mul/x:output:0=markov_chain_cnn_model/conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 26
4markov_chain_cnn_model/conv2d/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpG^markov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????``::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Fmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOpFmarkov_chain_cnn_model/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????``
 
_user_specified_nameinputs
?
?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_4356

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????``2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????``2
Relu?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02J
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SquareSquarePmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2;
9markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/SumSum=markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square:y:0Amarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum?
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *??82:
8markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x?
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mulMulAmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul/x:output:0?markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 28
6markov_chain_cnn_model/conv2d_5/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOpI^markov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????`` ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
Hmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOpHmarkov_chain_cnn_model/conv2d_5/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????`` 
 
_user_specified_nameinputs
?	
?
5__inference_markov_chain_cnn_model_layer_call_fn_5006
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????``*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_45682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????``2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????``::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????``
!
_user_specified_name	input_1"?L
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
serving_default_input_1:0?????????``D
output_18
StatefulPartitionedCall:0?????????``tensorflow/serving/predict:??
?
c1
c2
c3
maxpool

upsamp
c1_d
c2_d
c3_d
		variables

trainable_variables
regularization_losses
	keras_api

signatures
g__call__
*h&call_and_return_all_conditional_losses
i_default_save_signature"?
_tf_keras_model?{"class_name": "MarkovChainCNNModel", "name": "markov_chain_cnn_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MarkovChainCNNModel"}}
?


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 96, 96, 4]}}
?


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 48, 48, 16]}}
?


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 24, 24, 32]}}
?
 	variables
!regularization_losses
"trainable_variables
#	keras_api
p__call__
*q&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
$	variables
%regularization_losses
&trainable_variables
'	keras_api
r__call__
*s&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
t__call__
*u&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 48, 48, 96]}}
?


.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
v__call__
*w&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 96, 96, 80]}}
?


4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [64, 96, 96, 32]}}
v
0
1
2
3
4
5
(6
)7
.8
/9
410
511"
trackable_list_wrapper
v
0
1
2
3
4
5
(6
)7
.8
/9
410
511"
trackable_list_wrapper
J
z0
{1
|2
}3
~4
5"
trackable_list_wrapper
?
		variables

trainable_variables

:layers
regularization_losses
;layer_regularization_losses
<metrics
=layer_metrics
>non_trainable_variables
g__call__
i_default_save_signature
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
>:<2$markov_chain_cnn_model/conv2d/kernel
0:.2"markov_chain_cnn_model/conv2d/bias
.
0
1"
trackable_list_wrapper
'
z0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

?layers
@layer_regularization_losses
	variables
regularization_losses
trainable_variables
Ametrics
Blayer_metrics
Cnon_trainable_variables
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
@:> 2&markov_chain_cnn_model/conv2d_1/kernel
2:0 2$markov_chain_cnn_model/conv2d_1/bias
.
0
1"
trackable_list_wrapper
'
{0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Dlayers
Elayer_regularization_losses
	variables
regularization_losses
trainable_variables
Fmetrics
Glayer_metrics
Hnon_trainable_variables
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
@:> @2&markov_chain_cnn_model/conv2d_2/kernel
2:0@2$markov_chain_cnn_model/conv2d_2/bias
.
0
1"
trackable_list_wrapper
'
|0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Ilayers
Jlayer_regularization_losses
	variables
regularization_losses
trainable_variables
Kmetrics
Llayer_metrics
Mnon_trainable_variables
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Nlayers
Olayer_regularization_losses
 	variables
!regularization_losses
"trainable_variables
Pmetrics
Qlayer_metrics
Rnon_trainable_variables
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Slayers
Tlayer_regularization_losses
$	variables
%regularization_losses
&trainable_variables
Umetrics
Vlayer_metrics
Wnon_trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
@:>`@2&markov_chain_cnn_model/conv2d_3/kernel
2:0@2$markov_chain_cnn_model/conv2d_3/bias
.
(0
)1"
trackable_list_wrapper
'
}0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?

Xlayers
Ylayer_regularization_losses
*	variables
+regularization_losses
,trainable_variables
Zmetrics
[layer_metrics
\non_trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
@:>P 2&markov_chain_cnn_model/conv2d_4/kernel
2:0 2$markov_chain_cnn_model/conv2d_4/bias
.
.0
/1"
trackable_list_wrapper
'
~0"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?

]layers
^layer_regularization_losses
0	variables
1regularization_losses
2trainable_variables
_metrics
`layer_metrics
anon_trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
@:> 2&markov_chain_cnn_model/conv2d_5/kernel
2:02$markov_chain_cnn_model/conv2d_5/bias
.
40
51"
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?

blayers
clayer_regularization_losses
6	variables
7regularization_losses
8trainable_variables
dmetrics
elayer_metrics
fnon_trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
z0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
{0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
|0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
}0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
~0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
5__inference_markov_chain_cnn_model_layer_call_fn_5035
5__inference_markov_chain_cnn_model_layer_call_fn_5272
5__inference_markov_chain_cnn_model_layer_call_fn_5301
5__inference_markov_chain_cnn_model_layer_call_fn_5006?
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
?2?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_4873
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_4977
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_5139
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_5243?
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
__inference__wrapped_model_4131?
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
%__inference_conv2d_layer_call_fn_5333?
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
@__inference_conv2d_layer_call_and_return_conditional_losses_5324?
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
'__inference_conv2d_1_layer_call_fn_5365?
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
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5356?
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
'__inference_conv2d_2_layer_call_fn_5397?
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
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5388?
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
?2?
,__inference_max_pooling2d_layer_call_fn_4143?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4137?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_up_sampling2d_layer_call_fn_4162?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4156?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
'__inference_conv2d_3_layer_call_fn_5429?
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
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5420?
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
'__inference_conv2d_4_layer_call_fn_5461?
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
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5452?
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
'__inference_conv2d_5_layer_call_fn_5493?
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
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5484?
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
__inference_loss_fn_0_5504?
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
__inference_loss_fn_1_5515?
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
__inference_loss_fn_2_5526?
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
__inference_loss_fn_3_5537?
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
__inference_loss_fn_4_5548?
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
__inference_loss_fn_5_5559?
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
"__inference_signature_wrapper_4769input_1"?
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
__inference__wrapped_model_4131?()./458?5
.?+
)?&
input_1?????????``
? ";?8
6
output_1*?'
output_1?????????``?
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5356l7?4
-?*
(?%
inputs?????????00
? "-?*
#? 
0?????????00 
? ?
'__inference_conv2d_1_layer_call_fn_5365_7?4
-?*
(?%
inputs?????????00
? " ??????????00 ?
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5388l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
'__inference_conv2d_2_layer_call_fn_5397_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
B__inference_conv2d_3_layer_call_and_return_conditional_losses_5420l()7?4
-?*
(?%
inputs?????????00`
? "-?*
#? 
0?????????00@
? ?
'__inference_conv2d_3_layer_call_fn_5429_()7?4
-?*
(?%
inputs?????????00`
? " ??????????00@?
B__inference_conv2d_4_layer_call_and_return_conditional_losses_5452l./7?4
-?*
(?%
inputs?????????``P
? "-?*
#? 
0?????????`` 
? ?
'__inference_conv2d_4_layer_call_fn_5461_./7?4
-?*
(?%
inputs?????????``P
? " ??????????`` ?
B__inference_conv2d_5_layer_call_and_return_conditional_losses_5484l457?4
-?*
(?%
inputs?????????`` 
? "-?*
#? 
0?????????``
? ?
'__inference_conv2d_5_layer_call_fn_5493_457?4
-?*
(?%
inputs?????????`` 
? " ??????????``?
@__inference_conv2d_layer_call_and_return_conditional_losses_5324l7?4
-?*
(?%
inputs?????????``
? "-?*
#? 
0?????????``
? ?
%__inference_conv2d_layer_call_fn_5333_7?4
-?*
(?%
inputs?????????``
? " ??????????``9
__inference_loss_fn_0_5504?

? 
? "? 9
__inference_loss_fn_1_5515?

? 
? "? 9
__inference_loss_fn_2_5526?

? 
? "? 9
__inference_loss_fn_3_5537(?

? 
? "? 9
__inference_loss_fn_4_5548.?

? 
? "? 9
__inference_loss_fn_5_55594?

? 
? "? ?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_4873{()./45<?9
2?/
)?&
input_1?????????``
p
? "-?*
#? 
0?????????``
? ?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_4977{()./45<?9
2?/
)?&
input_1?????????``
p 
? "-?*
#? 
0?????????``
? ?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_5139y()./45:?7
0?-
'?$
input?????????``
p
? "-?*
#? 
0?????????``
? ?
P__inference_markov_chain_cnn_model_layer_call_and_return_conditional_losses_5243y()./45:?7
0?-
'?$
input?????????``
p 
? "-?*
#? 
0?????????``
? ?
5__inference_markov_chain_cnn_model_layer_call_fn_5006n()./45<?9
2?/
)?&
input_1?????????``
p
? " ??????????``?
5__inference_markov_chain_cnn_model_layer_call_fn_5035n()./45<?9
2?/
)?&
input_1?????????``
p 
? " ??????????``?
5__inference_markov_chain_cnn_model_layer_call_fn_5272l()./45:?7
0?-
'?$
input?????????``
p
? " ??????????``?
5__inference_markov_chain_cnn_model_layer_call_fn_5301l()./45:?7
0?-
'?$
input?????????``
p 
? " ??????????``?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4137?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_max_pooling2d_layer_call_fn_4143?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
"__inference_signature_wrapper_4769?()./45C?@
? 
9?6
4
input_1)?&
input_1?????????``";?8
6
output_1*?'
output_1?????????``?
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4156?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_up_sampling2d_layer_call_fn_4162?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????