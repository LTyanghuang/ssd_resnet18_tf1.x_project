??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%??8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
;
Maximum
x"T
y"T
z"T"
Ttype:

2	?
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
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
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
1
Square
x"T
y"T"
Ttype:

2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.13.12b'v1.13.0-rc2-5-g6612da8'??
p
input_imagePlaceholder*(
_output_shapes
:??*
shape:??*
dtype0
?
ssd_300_vgg/Pad/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
?
ssd_300_vgg/PadPadinput_imagessd_300_vgg/Pad/paddings*
	Tpaddings0*
T0*(
_output_shapes
:??
?
>ssd_300_vgg/conv_init/weights/Initializer/random_uniform/shapeConst*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*
_output_shapes
:*
dtype0*%
valueB"         @   
?
<ssd_300_vgg/conv_init/weights/Initializer/random_uniform/minConst*
dtype0*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*
_output_shapes
: *
valueB
 */?
?
<ssd_300_vgg/conv_init/weights/Initializer/random_uniform/maxConst*
valueB
 */=*
dtype0*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*
_output_shapes
: 
?
Fssd_300_vgg/conv_init/weights/Initializer/random_uniform/RandomUniformRandomUniform>ssd_300_vgg/conv_init/weights/Initializer/random_uniform/shape*&
_output_shapes
:@*
seed2 *
dtype0*
T0*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*

seed 
?
<ssd_300_vgg/conv_init/weights/Initializer/random_uniform/subSub<ssd_300_vgg/conv_init/weights/Initializer/random_uniform/max<ssd_300_vgg/conv_init/weights/Initializer/random_uniform/min*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*
T0*
_output_shapes
: 
?
<ssd_300_vgg/conv_init/weights/Initializer/random_uniform/mulMulFssd_300_vgg/conv_init/weights/Initializer/random_uniform/RandomUniform<ssd_300_vgg/conv_init/weights/Initializer/random_uniform/sub*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*&
_output_shapes
:@*
T0
?
8ssd_300_vgg/conv_init/weights/Initializer/random_uniformAdd<ssd_300_vgg/conv_init/weights/Initializer/random_uniform/mul<ssd_300_vgg/conv_init/weights/Initializer/random_uniform/min*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*
T0*&
_output_shapes
:@
?
ssd_300_vgg/conv_init/weights
VariableV2*
shape:@*
dtype0*
	container *&
_output_shapes
:@*
shared_name *0
_class&
$"loc:@ssd_300_vgg/conv_init/weights
?
$ssd_300_vgg/conv_init/weights/AssignAssignssd_300_vgg/conv_init/weights8ssd_300_vgg/conv_init/weights/Initializer/random_uniform*
T0*
use_locking(*&
_output_shapes
:@*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*
validate_shape(
?
"ssd_300_vgg/conv_init/weights/readIdentityssd_300_vgg/conv_init/weights*&
_output_shapes
:@*
T0*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights
?
=ssd_300_vgg/conv_init/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:
?
>ssd_300_vgg/conv_init/kernel/Regularizer/l2_regularizer/L2LossL2Loss"ssd_300_vgg/conv_init/weights/read*
_output_shapes
: *
T0
?
7ssd_300_vgg/conv_init/kernel/Regularizer/l2_regularizerMul=ssd_300_vgg/conv_init/kernel/Regularizer/l2_regularizer/scale>ssd_300_vgg/conv_init/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
.ssd_300_vgg/conv_init/biases/Initializer/zerosConst*/
_class%
#!loc:@ssd_300_vgg/conv_init/biases*
dtype0*
valueB@*    *
_output_shapes
:@
?
ssd_300_vgg/conv_init/biases
VariableV2*/
_class%
#!loc:@ssd_300_vgg/conv_init/biases*
shared_name *
dtype0*
	container *
shape:@*
_output_shapes
:@
?
#ssd_300_vgg/conv_init/biases/AssignAssignssd_300_vgg/conv_init/biases.ssd_300_vgg/conv_init/biases/Initializer/zeros*
validate_shape(*
T0*/
_class%
#!loc:@ssd_300_vgg/conv_init/biases*
_output_shapes
:@*
use_locking(
?
!ssd_300_vgg/conv_init/biases/readIdentityssd_300_vgg/conv_init/biases*/
_class%
#!loc:@ssd_300_vgg/conv_init/biases*
_output_shapes
:@*
T0
t
#ssd_300_vgg/conv_init/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
ssd_300_vgg/conv_init/Conv2DConv2Dssd_300_vgg/Pad"ssd_300_vgg/conv_init/weights/read*
T0*
data_formatNHWC*
paddingVALID*
use_cudnn_on_gpu(*&
_output_shapes
:pp@*
strides
*
	dilations

?
ssd_300_vgg/conv_init/BiasAddBiasAddssd_300_vgg/conv_init/Conv2D!ssd_300_vgg/conv_init/biases/read*
data_formatNHWC*
T0*&
_output_shapes
:pp@
?
0ssd_300_vgg/batch_norm_00/beta/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *
dtype0*1
_class'
%#loc:@ssd_300_vgg/batch_norm_00/beta
?
ssd_300_vgg/batch_norm_00/beta
VariableV2*
shape:@*
_output_shapes
:@*1
_class'
%#loc:@ssd_300_vgg/batch_norm_00/beta*
shared_name *
dtype0*
	container 
?
%ssd_300_vgg/batch_norm_00/beta/AssignAssignssd_300_vgg/batch_norm_00/beta0ssd_300_vgg/batch_norm_00/beta/Initializer/zeros*
T0*
validate_shape(*1
_class'
%#loc:@ssd_300_vgg/batch_norm_00/beta*
_output_shapes
:@*
use_locking(
?
#ssd_300_vgg/batch_norm_00/beta/readIdentityssd_300_vgg/batch_norm_00/beta*
T0*
_output_shapes
:@*1
_class'
%#loc:@ssd_300_vgg/batch_norm_00/beta
?
0ssd_300_vgg/batch_norm_00/gamma/Initializer/onesConst*2
_class(
&$loc:@ssd_300_vgg/batch_norm_00/gamma*
valueB@*  ??*
_output_shapes
:@*
dtype0
?
ssd_300_vgg/batch_norm_00/gamma
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*2
_class(
&$loc:@ssd_300_vgg/batch_norm_00/gamma*
shared_name 
?
&ssd_300_vgg/batch_norm_00/gamma/AssignAssignssd_300_vgg/batch_norm_00/gamma0ssd_300_vgg/batch_norm_00/gamma/Initializer/ones*
_output_shapes
:@*
T0*
use_locking(*2
_class(
&$loc:@ssd_300_vgg/batch_norm_00/gamma*
validate_shape(
?
$ssd_300_vgg/batch_norm_00/gamma/readIdentityssd_300_vgg/batch_norm_00/gamma*
_output_shapes
:@*2
_class(
&$loc:@ssd_300_vgg/batch_norm_00/gamma*
T0
?
7ssd_300_vgg/batch_norm_00/moving_mean/Initializer/zerosConst*
_output_shapes
:@*8
_class.
,*loc:@ssd_300_vgg/batch_norm_00/moving_mean*
valueB@*    *
dtype0
?
%ssd_300_vgg/batch_norm_00/moving_mean
VariableV2*
dtype0*
	container *
shared_name *
shape:@*
_output_shapes
:@*8
_class.
,*loc:@ssd_300_vgg/batch_norm_00/moving_mean
?
,ssd_300_vgg/batch_norm_00/moving_mean/AssignAssign%ssd_300_vgg/batch_norm_00/moving_mean7ssd_300_vgg/batch_norm_00/moving_mean/Initializer/zeros*
T0*
use_locking(*8
_class.
,*loc:@ssd_300_vgg/batch_norm_00/moving_mean*
_output_shapes
:@*
validate_shape(
?
*ssd_300_vgg/batch_norm_00/moving_mean/readIdentity%ssd_300_vgg/batch_norm_00/moving_mean*
_output_shapes
:@*
T0*8
_class.
,*loc:@ssd_300_vgg/batch_norm_00/moving_mean
?
:ssd_300_vgg/batch_norm_00/moving_variance/Initializer/onesConst*
dtype0*<
_class2
0.loc:@ssd_300_vgg/batch_norm_00/moving_variance*
_output_shapes
:@*
valueB@*  ??
?
)ssd_300_vgg/batch_norm_00/moving_variance
VariableV2*
dtype0*<
_class2
0.loc:@ssd_300_vgg/batch_norm_00/moving_variance*
_output_shapes
:@*
shape:@*
	container *
shared_name 
?
0ssd_300_vgg/batch_norm_00/moving_variance/AssignAssign)ssd_300_vgg/batch_norm_00/moving_variance:ssd_300_vgg/batch_norm_00/moving_variance/Initializer/ones*
T0*
use_locking(*
_output_shapes
:@*<
_class2
0.loc:@ssd_300_vgg/batch_norm_00/moving_variance*
validate_shape(
?
.ssd_300_vgg/batch_norm_00/moving_variance/readIdentity)ssd_300_vgg/batch_norm_00/moving_variance*<
_class2
0.loc:@ssd_300_vgg/batch_norm_00/moving_variance*
_output_shapes
:@*
T0
?
(ssd_300_vgg/batch_norm_00/FusedBatchNormFusedBatchNormssd_300_vgg/conv_init/BiasAdd$ssd_300_vgg/batch_norm_00/gamma/read#ssd_300_vgg/batch_norm_00/beta/read*ssd_300_vgg/batch_norm_00/moving_mean/read.ssd_300_vgg/batch_norm_00/moving_variance/read*
data_formatNHWC*
T0*>
_output_shapes,
*:pp@:@:@:@:@*
is_training( *
epsilon%??'7
s
ssd_300_vgg/ReluRelu(ssd_300_vgg/batch_norm_00/FusedBatchNorm*
T0*&
_output_shapes
:pp@
?
ssd_300_vgg/maxpool_0/MaxPoolMaxPoolssd_300_vgg/Relu*
ksize
*
paddingSAME*
T0*
strides
*&
_output_shapes
:88@*
data_formatNHWC
?
@ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights*
dtype0*
_output_shapes
:*%
valueB"      @   @   
?
>ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/minConst*
dtype0*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights*
_output_shapes
: *
valueB
 *:͓?
?
>ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *:͓=*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights*
dtype0
?
Hssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/RandomUniformRandomUniform@ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/shape*
seed2 *
dtype0*
T0*

seed *2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights*&
_output_shapes
:@@
?
>ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/subSub>ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/max>ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/min*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights*
T0*
_output_shapes
: 
?
>ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/mulMulHssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/RandomUniform>ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/sub*&
_output_shapes
:@@*
T0*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights
?
:ssd_300_vgg/conv_init_1/weights/Initializer/random_uniformAdd>ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/mul>ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights*&
_output_shapes
:@@
?
ssd_300_vgg/conv_init_1/weights
VariableV2*
	container *
shape:@@*&
_output_shapes
:@@*
dtype0*
shared_name *2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights
?
&ssd_300_vgg/conv_init_1/weights/AssignAssignssd_300_vgg/conv_init_1/weights:ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform*
validate_shape(*
use_locking(*&
_output_shapes
:@@*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights*
T0
?
$ssd_300_vgg/conv_init_1/weights/readIdentityssd_300_vgg/conv_init_1/weights*
T0*&
_output_shapes
:@@*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights
?
?ssd_300_vgg/conv_init_1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *o:*
dtype0
?
@ssd_300_vgg/conv_init_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss$ssd_300_vgg/conv_init_1/weights/read*
_output_shapes
: *
T0
?
9ssd_300_vgg/conv_init_1/kernel/Regularizer/l2_regularizerMul?ssd_300_vgg/conv_init_1/kernel/Regularizer/l2_regularizer/scale@ssd_300_vgg/conv_init_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
0ssd_300_vgg/conv_init_1/biases/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *1
_class'
%#loc:@ssd_300_vgg/conv_init_1/biases*
dtype0
?
ssd_300_vgg/conv_init_1/biases
VariableV2*1
_class'
%#loc:@ssd_300_vgg/conv_init_1/biases*
_output_shapes
:@*
	container *
shared_name *
dtype0*
shape:@
?
%ssd_300_vgg/conv_init_1/biases/AssignAssignssd_300_vgg/conv_init_1/biases0ssd_300_vgg/conv_init_1/biases/Initializer/zeros*
_output_shapes
:@*
use_locking(*
validate_shape(*1
_class'
%#loc:@ssd_300_vgg/conv_init_1/biases*
T0
?
#ssd_300_vgg/conv_init_1/biases/readIdentityssd_300_vgg/conv_init_1/biases*
T0*1
_class'
%#loc:@ssd_300_vgg/conv_init_1/biases*
_output_shapes
:@
v
%ssd_300_vgg/conv_init_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
ssd_300_vgg/conv_init_1/Conv2DConv2Dssd_300_vgg/maxpool_0/MaxPool$ssd_300_vgg/conv_init_1/weights/read*
T0*
strides
*
paddingSAME*
	dilations
*&
_output_shapes
:88@*
data_formatNHWC*
use_cudnn_on_gpu(
?
ssd_300_vgg/conv_init_1/BiasAddBiasAddssd_300_vgg/conv_init_1/Conv2D#ssd_300_vgg/conv_init_1/biases/read*&
_output_shapes
:88@*
data_formatNHWC*
T0
?
;ssd_300_vgg/resblock0_0/batch_norm_0/beta/Initializer/zerosConst*
_output_shapes
:@*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_0/beta*
valueB@*    *
dtype0
?
)ssd_300_vgg/resblock0_0/batch_norm_0/beta
VariableV2*
	container *
_output_shapes
:@*
shared_name *
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_0/beta*
shape:@
?
0ssd_300_vgg/resblock0_0/batch_norm_0/beta/AssignAssign)ssd_300_vgg/resblock0_0/batch_norm_0/beta;ssd_300_vgg/resblock0_0/batch_norm_0/beta/Initializer/zeros*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_0/beta*
validate_shape(*
_output_shapes
:@*
T0
?
.ssd_300_vgg/resblock0_0/batch_norm_0/beta/readIdentity)ssd_300_vgg/resblock0_0/batch_norm_0/beta*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_0/beta*
_output_shapes
:@
?
;ssd_300_vgg/resblock0_0/batch_norm_0/gamma/Initializer/onesConst*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_0/gamma*
valueB@*  ??*
_output_shapes
:@*
dtype0
?
*ssd_300_vgg/resblock0_0/batch_norm_0/gamma
VariableV2*
	container *
shared_name *=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_0/gamma*
shape:@*
dtype0*
_output_shapes
:@
?
1ssd_300_vgg/resblock0_0/batch_norm_0/gamma/AssignAssign*ssd_300_vgg/resblock0_0/batch_norm_0/gamma;ssd_300_vgg/resblock0_0/batch_norm_0/gamma/Initializer/ones*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_0/gamma*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
?
/ssd_300_vgg/resblock0_0/batch_norm_0/gamma/readIdentity*ssd_300_vgg/resblock0_0/batch_norm_0/gamma*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_0/gamma*
_output_shapes
:@*
T0
?
Bssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean
?
0ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean
VariableV2*
shape:@*
dtype0*
	container *C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean*
shared_name *
_output_shapes
:@
?
7ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/AssignAssign0ssd_300_vgg/resblock0_0/batch_norm_0/moving_meanBssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/Initializer/zeros*C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@
?
5ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/readIdentity0ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean*
T0*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean
?
Essd_300_vgg/resblock0_0/batch_norm_0/moving_variance/Initializer/onesConst*
_output_shapes
:@*
valueB@*  ??*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance*
dtype0
?
4ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance
VariableV2*
shape:@*
	container *
dtype0*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance*
_output_shapes
:@*
shared_name 
?
;ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/AssignAssign4ssd_300_vgg/resblock0_0/batch_norm_0/moving_varianceEssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/Initializer/ones*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance*
validate_shape(*
_output_shapes
:@*
use_locking(
?
9ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/readIdentity4ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance*
T0*
_output_shapes
:@
?
3ssd_300_vgg/resblock0_0/batch_norm_0/FusedBatchNormFusedBatchNormssd_300_vgg/conv_init_1/BiasAdd/ssd_300_vgg/resblock0_0/batch_norm_0/gamma/read.ssd_300_vgg/resblock0_0/batch_norm_0/beta/read5ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/read9ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/read*
T0*>
_output_shapes,
*:88@:@:@:@:@*
is_training( *
data_formatNHWC*
epsilon%??'7
?
ssd_300_vgg/resblock0_0/ReluRelu3ssd_300_vgg/resblock0_0/batch_norm_0/FusedBatchNorm*&
_output_shapes
:88@*
T0
?
Gssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
_output_shapes
:*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights
?
Essd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/minConst*
valueB
 *:͓?*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights*
dtype0
?
Essd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/maxConst*
valueB
 *:͓=*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights*
dtype0
?
Ossd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/shape*&
_output_shapes
:@@*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights*
dtype0*
T0*

seed *
seed2 
?
Essd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/min*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights*
T0
?
Essd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/sub*
T0*&
_output_shapes
:@@*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights
?
Assd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniformAddEssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform/min*&
_output_shapes
:@@*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights*
T0
?
&ssd_300_vgg/resblock0_0/conv_0/weights
VariableV2*
shape:@@*
dtype0*
shared_name *9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights*&
_output_shapes
:@@*
	container 
?
-ssd_300_vgg/resblock0_0/conv_0/weights/AssignAssign&ssd_300_vgg/resblock0_0/conv_0/weightsAssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform*
T0*&
_output_shapes
:@@*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights*
use_locking(*
validate_shape(
?
+ssd_300_vgg/resblock0_0/conv_0/weights/readIdentity&ssd_300_vgg/resblock0_0/conv_0/weights*&
_output_shapes
:@@*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights*
T0
?
Fssd_300_vgg/resblock0_0/conv_0/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:
?
Gssd_300_vgg/resblock0_0/conv_0/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock0_0/conv_0/weights/read*
T0*
_output_shapes
: 
?
@ssd_300_vgg/resblock0_0/conv_0/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock0_0/conv_0/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock0_0/conv_0/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
7ssd_300_vgg/resblock0_0/conv_0/biases/Initializer/zerosConst*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_0/biases*
_output_shapes
:@*
dtype0*
valueB@*    
?
%ssd_300_vgg/resblock0_0/conv_0/biases
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_0/biases*
shape:@
?
,ssd_300_vgg/resblock0_0/conv_0/biases/AssignAssign%ssd_300_vgg/resblock0_0/conv_0/biases7ssd_300_vgg/resblock0_0/conv_0/biases/Initializer/zeros*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_0/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
?
*ssd_300_vgg/resblock0_0/conv_0/biases/readIdentity%ssd_300_vgg/resblock0_0/conv_0/biases*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_0/biases*
T0*
_output_shapes
:@
}
,ssd_300_vgg/resblock0_0/conv_0/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
%ssd_300_vgg/resblock0_0/conv_0/Conv2DConv2Dssd_300_vgg/resblock0_0/Relu+ssd_300_vgg/resblock0_0/conv_0/weights/read*&
_output_shapes
:88@*
T0*
strides
*
data_formatNHWC*
	dilations
*
paddingSAME*
use_cudnn_on_gpu(
?
&ssd_300_vgg/resblock0_0/conv_0/BiasAddBiasAdd%ssd_300_vgg/resblock0_0/conv_0/Conv2D*ssd_300_vgg/resblock0_0/conv_0/biases/read*
T0*&
_output_shapes
:88@*
data_formatNHWC
?
;ssd_300_vgg/resblock0_0/batch_norm_1/beta/Initializer/zerosConst*
valueB@*    *
dtype0*
_output_shapes
:@*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_1/beta
?
)ssd_300_vgg/resblock0_0/batch_norm_1/beta
VariableV2*
	container *
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_1/beta*
shared_name *
shape:@*
_output_shapes
:@
?
0ssd_300_vgg/resblock0_0/batch_norm_1/beta/AssignAssign)ssd_300_vgg/resblock0_0/batch_norm_1/beta;ssd_300_vgg/resblock0_0/batch_norm_1/beta/Initializer/zeros*
_output_shapes
:@*
validate_shape(*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_1/beta*
T0*
use_locking(
?
.ssd_300_vgg/resblock0_0/batch_norm_1/beta/readIdentity)ssd_300_vgg/resblock0_0/batch_norm_1/beta*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_1/beta*
_output_shapes
:@
?
;ssd_300_vgg/resblock0_0/batch_norm_1/gamma/Initializer/onesConst*
valueB@*  ??*
_output_shapes
:@*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_1/gamma*
dtype0
?
*ssd_300_vgg/resblock0_0/batch_norm_1/gamma
VariableV2*
dtype0*
shared_name *
_output_shapes
:@*
shape:@*
	container *=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_1/gamma
?
1ssd_300_vgg/resblock0_0/batch_norm_1/gamma/AssignAssign*ssd_300_vgg/resblock0_0/batch_norm_1/gamma;ssd_300_vgg/resblock0_0/batch_norm_1/gamma/Initializer/ones*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_1/gamma*
use_locking(*
_output_shapes
:@*
validate_shape(
?
/ssd_300_vgg/resblock0_0/batch_norm_1/gamma/readIdentity*ssd_300_vgg/resblock0_0/batch_norm_1/gamma*
T0*
_output_shapes
:@*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_1/gamma
?
Bssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/Initializer/zerosConst*
valueB@*    *C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean*
_output_shapes
:@*
dtype0
?
0ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean
VariableV2*C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean*
	container *
shape:@*
shared_name *
_output_shapes
:@*
dtype0
?
7ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/AssignAssign0ssd_300_vgg/resblock0_0/batch_norm_1/moving_meanBssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/Initializer/zeros*
validate_shape(*
use_locking(*C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean*
T0*
_output_shapes
:@
?
5ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/readIdentity0ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean*
T0*C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean*
_output_shapes
:@
?
Essd_300_vgg/resblock0_0/batch_norm_1/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:@*
valueB@*  ??*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance
?
4ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance
VariableV2*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance*
shape:@*
shared_name *
_output_shapes
:@*
	container *
dtype0
?
;ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/AssignAssign4ssd_300_vgg/resblock0_0/batch_norm_1/moving_varianceEssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/Initializer/ones*
_output_shapes
:@*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance*
validate_shape(*
use_locking(*
T0
?
9ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/readIdentity4ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance*
_output_shapes
:@
?
3ssd_300_vgg/resblock0_0/batch_norm_1/FusedBatchNormFusedBatchNorm&ssd_300_vgg/resblock0_0/conv_0/BiasAdd/ssd_300_vgg/resblock0_0/batch_norm_1/gamma/read.ssd_300_vgg/resblock0_0/batch_norm_1/beta/read5ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/read9ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/read*>
_output_shapes,
*:88@:@:@:@:@*
is_training( *
epsilon%??'7*
T0*
data_formatNHWC
?
ssd_300_vgg/resblock0_0/Relu_1Relu3ssd_300_vgg/resblock0_0/batch_norm_1/FusedBatchNorm*&
_output_shapes
:88@*
T0
?
Gssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"      @   @   *9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*
dtype0
?
Essd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/minConst*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*
valueB
 *:͓?*
_output_shapes
: 
?
Essd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/maxConst*
valueB
 *:͓=*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*
dtype0
?
Ossd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/shape*

seed *
dtype0*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*&
_output_shapes
:@@*
seed2 
?
Essd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights
?
Essd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/sub*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*&
_output_shapes
:@@
?
Assd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniformAddEssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*
T0*&
_output_shapes
:@@
?
&ssd_300_vgg/resblock0_0/conv_1/weights
VariableV2*
shape:@@*&
_output_shapes
:@@*
shared_name *
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*
	container 
?
-ssd_300_vgg/resblock0_0/conv_1/weights/AssignAssign&ssd_300_vgg/resblock0_0/conv_1/weightsAssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*
T0*
validate_shape(*&
_output_shapes
:@@
?
+ssd_300_vgg/resblock0_0/conv_1/weights/readIdentity&ssd_300_vgg/resblock0_0/conv_1/weights*&
_output_shapes
:@@*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights
?
Fssd_300_vgg/resblock0_0/conv_1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Gssd_300_vgg/resblock0_0/conv_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock0_0/conv_1/weights/read*
T0*
_output_shapes
: 
?
@ssd_300_vgg/resblock0_0/conv_1/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock0_0/conv_1/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock0_0/conv_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
7ssd_300_vgg/resblock0_0/conv_1/biases/Initializer/zerosConst*
valueB@*    *8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_1/biases*
_output_shapes
:@*
dtype0
?
%ssd_300_vgg/resblock0_0/conv_1/biases
VariableV2*
_output_shapes
:@*
dtype0*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_1/biases*
shared_name *
	container *
shape:@
?
,ssd_300_vgg/resblock0_0/conv_1/biases/AssignAssign%ssd_300_vgg/resblock0_0/conv_1/biases7ssd_300_vgg/resblock0_0/conv_1/biases/Initializer/zeros*
validate_shape(*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_1/biases*
_output_shapes
:@*
use_locking(
?
*ssd_300_vgg/resblock0_0/conv_1/biases/readIdentity%ssd_300_vgg/resblock0_0/conv_1/biases*
_output_shapes
:@*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_1/biases*
T0
}
,ssd_300_vgg/resblock0_0/conv_1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
%ssd_300_vgg/resblock0_0/conv_1/Conv2DConv2Dssd_300_vgg/resblock0_0/Relu_1+ssd_300_vgg/resblock0_0/conv_1/weights/read*
data_formatNHWC*
	dilations
*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
T0*&
_output_shapes
:88@
?
&ssd_300_vgg/resblock0_0/conv_1/BiasAddBiasAdd%ssd_300_vgg/resblock0_0/conv_1/Conv2D*ssd_300_vgg/resblock0_0/conv_1/biases/read*&
_output_shapes
:88@*
data_formatNHWC*
T0
?
ssd_300_vgg/resblock0_0/addAdd&ssd_300_vgg/resblock0_0/conv_1/BiasAddssd_300_vgg/conv_init_1/BiasAdd*&
_output_shapes
:88@*
T0
?
;ssd_300_vgg/resblock0_1/batch_norm_0/beta/Initializer/zerosConst*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_0/beta*
valueB@*    *
dtype0*
_output_shapes
:@
?
)ssd_300_vgg/resblock0_1/batch_norm_0/beta
VariableV2*
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_0/beta*
	container *
shape:@*
shared_name *
_output_shapes
:@
?
0ssd_300_vgg/resblock0_1/batch_norm_0/beta/AssignAssign)ssd_300_vgg/resblock0_1/batch_norm_0/beta;ssd_300_vgg/resblock0_1/batch_norm_0/beta/Initializer/zeros*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_0/beta*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@
?
.ssd_300_vgg/resblock0_1/batch_norm_0/beta/readIdentity)ssd_300_vgg/resblock0_1/batch_norm_0/beta*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_0/beta*
T0*
_output_shapes
:@
?
;ssd_300_vgg/resblock0_1/batch_norm_0/gamma/Initializer/onesConst*
dtype0*
valueB@*  ??*
_output_shapes
:@*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_0/gamma
?
*ssd_300_vgg/resblock0_1/batch_norm_0/gamma
VariableV2*
shared_name *
shape:@*
dtype0*
_output_shapes
:@*
	container *=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_0/gamma
?
1ssd_300_vgg/resblock0_1/batch_norm_0/gamma/AssignAssign*ssd_300_vgg/resblock0_1/batch_norm_0/gamma;ssd_300_vgg/resblock0_1/batch_norm_0/gamma/Initializer/ones*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_0/gamma*
use_locking(*
validate_shape(*
_output_shapes
:@
?
/ssd_300_vgg/resblock0_1/batch_norm_0/gamma/readIdentity*ssd_300_vgg/resblock0_1/batch_norm_0/gamma*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_0/gamma*
T0*
_output_shapes
:@
?
Bssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/Initializer/zerosConst*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean*
valueB@*    *
dtype0
?
0ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean
VariableV2*
dtype0*
shared_name *
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean*
shape:@*
	container 
?
7ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/AssignAssign0ssd_300_vgg/resblock0_1/batch_norm_0/moving_meanBssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/Initializer/zeros*
validate_shape(*
use_locking(*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean*
T0*
_output_shapes
:@
?
5ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/readIdentity0ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean*
T0
?
Essd_300_vgg/resblock0_1/batch_norm_0/moving_variance/Initializer/onesConst*
_output_shapes
:@*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance*
dtype0*
valueB@*  ??
?
4ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *
shape:@*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance*
	container 
?
;ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/AssignAssign4ssd_300_vgg/resblock0_1/batch_norm_0/moving_varianceEssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/Initializer/ones*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance
?
9ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/readIdentity4ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance*
T0*
_output_shapes
:@*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance
?
3ssd_300_vgg/resblock0_1/batch_norm_0/FusedBatchNormFusedBatchNormssd_300_vgg/resblock0_0/add/ssd_300_vgg/resblock0_1/batch_norm_0/gamma/read.ssd_300_vgg/resblock0_1/batch_norm_0/beta/read5ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/read9ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/read*
is_training( *>
_output_shapes,
*:88@:@:@:@:@*
epsilon%??'7*
data_formatNHWC*
T0
?
ssd_300_vgg/resblock0_1/ReluRelu3ssd_300_vgg/resblock0_1/batch_norm_0/FusedBatchNorm*
T0*&
_output_shapes
:88@
?
Gssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights*
dtype0*%
valueB"      @   @   
?
Essd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:͓?*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights
?
Essd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/maxConst*
valueB
 *:͓=*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights*
dtype0
?
Ossd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/shape*

seed *
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights*&
_output_shapes
:@@*
seed2 *
dtype0
?
Essd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights
?
Essd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights*&
_output_shapes
:@@*
T0
?
Assd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniformAddEssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform/min*&
_output_shapes
:@@*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights
?
&ssd_300_vgg/resblock0_1/conv_0/weights
VariableV2*
shape:@@*
	container *
shared_name *9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights*&
_output_shapes
:@@*
dtype0
?
-ssd_300_vgg/resblock0_1/conv_0/weights/AssignAssign&ssd_300_vgg/resblock0_1/conv_0/weightsAssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform*
T0*&
_output_shapes
:@@*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights*
validate_shape(*
use_locking(
?
+ssd_300_vgg/resblock0_1/conv_0/weights/readIdentity&ssd_300_vgg/resblock0_1/conv_0/weights*
T0*&
_output_shapes
:@@*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights
?
Fssd_300_vgg/resblock0_1/conv_0/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
_output_shapes
: *
dtype0
?
Gssd_300_vgg/resblock0_1/conv_0/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock0_1/conv_0/weights/read*
_output_shapes
: *
T0
?
@ssd_300_vgg/resblock0_1/conv_0/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock0_1/conv_0/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock0_1/conv_0/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
7ssd_300_vgg/resblock0_1/conv_0/biases/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_0/biases*
_output_shapes
:@*
valueB@*    
?
%ssd_300_vgg/resblock0_1/conv_0/biases
VariableV2*
shared_name *
dtype0*
	container *8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_0/biases*
shape:@*
_output_shapes
:@
?
,ssd_300_vgg/resblock0_1/conv_0/biases/AssignAssign%ssd_300_vgg/resblock0_1/conv_0/biases7ssd_300_vgg/resblock0_1/conv_0/biases/Initializer/zeros*
T0*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_0/biases*
use_locking(*
_output_shapes
:@
?
*ssd_300_vgg/resblock0_1/conv_0/biases/readIdentity%ssd_300_vgg/resblock0_1/conv_0/biases*
_output_shapes
:@*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_0/biases
}
,ssd_300_vgg/resblock0_1/conv_0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
%ssd_300_vgg/resblock0_1/conv_0/Conv2DConv2Dssd_300_vgg/resblock0_1/Relu+ssd_300_vgg/resblock0_1/conv_0/weights/read*
T0*
paddingSAME*
use_cudnn_on_gpu(*
data_formatNHWC*
	dilations
*&
_output_shapes
:88@*
strides

?
&ssd_300_vgg/resblock0_1/conv_0/BiasAddBiasAdd%ssd_300_vgg/resblock0_1/conv_0/Conv2D*ssd_300_vgg/resblock0_1/conv_0/biases/read*
T0*
data_formatNHWC*&
_output_shapes
:88@
?
;ssd_300_vgg/resblock0_1/batch_norm_1/beta/Initializer/zerosConst*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_1/beta*
dtype0*
valueB@*    *
_output_shapes
:@
?
)ssd_300_vgg/resblock0_1/batch_norm_1/beta
VariableV2*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_1/beta*
	container *
_output_shapes
:@*
shape:@*
dtype0*
shared_name 
?
0ssd_300_vgg/resblock0_1/batch_norm_1/beta/AssignAssign)ssd_300_vgg/resblock0_1/batch_norm_1/beta;ssd_300_vgg/resblock0_1/batch_norm_1/beta/Initializer/zeros*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_1/beta
?
.ssd_300_vgg/resblock0_1/batch_norm_1/beta/readIdentity)ssd_300_vgg/resblock0_1/batch_norm_1/beta*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_1/beta*
_output_shapes
:@*
T0
?
;ssd_300_vgg/resblock0_1/batch_norm_1/gamma/Initializer/onesConst*
valueB@*  ??*
dtype0*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_1/gamma*
_output_shapes
:@
?
*ssd_300_vgg/resblock0_1/batch_norm_1/gamma
VariableV2*
dtype0*
	container *=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_1/gamma*
shared_name *
shape:@*
_output_shapes
:@
?
1ssd_300_vgg/resblock0_1/batch_norm_1/gamma/AssignAssign*ssd_300_vgg/resblock0_1/batch_norm_1/gamma;ssd_300_vgg/resblock0_1/batch_norm_1/gamma/Initializer/ones*
_output_shapes
:@*
use_locking(*
validate_shape(*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_1/gamma
?
/ssd_300_vgg/resblock0_1/batch_norm_1/gamma/readIdentity*ssd_300_vgg/resblock0_1/batch_norm_1/gamma*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_1/gamma*
_output_shapes
:@
?
Bssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/Initializer/zerosConst*
_output_shapes
:@*
dtype0*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean*
valueB@*    
?
0ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean
VariableV2*
_output_shapes
:@*
shared_name *C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean*
dtype0*
	container *
shape:@
?
7ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/AssignAssign0ssd_300_vgg/resblock0_1/batch_norm_1/moving_meanBssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/Initializer/zeros*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean
?
5ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/readIdentity0ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean*
T0*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean
?
Essd_300_vgg/resblock0_1/batch_norm_1/moving_variance/Initializer/onesConst*
valueB@*  ??*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance*
dtype0*
_output_shapes
:@
?
4ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance
VariableV2*
	container *
shape:@*
dtype0*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance*
shared_name *
_output_shapes
:@
?
;ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/AssignAssign4ssd_300_vgg/resblock0_1/batch_norm_1/moving_varianceEssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/Initializer/ones*
_output_shapes
:@*
use_locking(*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance*
validate_shape(*
T0
?
9ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/readIdentity4ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance*
_output_shapes
:@*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance*
T0
?
3ssd_300_vgg/resblock0_1/batch_norm_1/FusedBatchNormFusedBatchNorm&ssd_300_vgg/resblock0_1/conv_0/BiasAdd/ssd_300_vgg/resblock0_1/batch_norm_1/gamma/read.ssd_300_vgg/resblock0_1/batch_norm_1/beta/read5ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/read9ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/read*
is_training( *
epsilon%??'7*
T0*>
_output_shapes,
*:88@:@:@:@:@*
data_formatNHWC
?
ssd_300_vgg/resblock0_1/Relu_1Relu3ssd_300_vgg/resblock0_1/batch_norm_1/FusedBatchNorm*&
_output_shapes
:88@*
T0
?
Gssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/shapeConst*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*%
valueB"      @   @   *
_output_shapes
:*
dtype0
?
Essd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/minConst*
dtype0*
valueB
 *:͓?*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*
_output_shapes
: 
?
Essd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *:͓=*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights
?
Ossd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/shape*
seed2 *&
_output_shapes
:@@*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*
dtype0*

seed 
?
Essd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*
_output_shapes
: 
?
Essd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*&
_output_shapes
:@@*
T0
?
Assd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniformAddEssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*&
_output_shapes
:@@*
T0
?
&ssd_300_vgg/resblock0_1/conv_1/weights
VariableV2*
shape:@@*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*&
_output_shapes
:@@*
shared_name *
	container 
?
-ssd_300_vgg/resblock0_1/conv_1/weights/AssignAssign&ssd_300_vgg/resblock0_1/conv_1/weightsAssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*
validate_shape(*
T0*&
_output_shapes
:@@*
use_locking(
?
+ssd_300_vgg/resblock0_1/conv_1/weights/readIdentity&ssd_300_vgg/resblock0_1/conv_1/weights*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*
T0*&
_output_shapes
:@@
?
Fssd_300_vgg/resblock0_1/conv_1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Gssd_300_vgg/resblock0_1/conv_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock0_1/conv_1/weights/read*
_output_shapes
: *
T0
?
@ssd_300_vgg/resblock0_1/conv_1/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock0_1/conv_1/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock0_1/conv_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
7ssd_300_vgg/resblock0_1/conv_1/biases/Initializer/zerosConst*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_1/biases*
valueB@*    *
dtype0*
_output_shapes
:@
?
%ssd_300_vgg/resblock0_1/conv_1/biases
VariableV2*
dtype0*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_1/biases*
shared_name *
	container *
_output_shapes
:@*
shape:@
?
,ssd_300_vgg/resblock0_1/conv_1/biases/AssignAssign%ssd_300_vgg/resblock0_1/conv_1/biases7ssd_300_vgg/resblock0_1/conv_1/biases/Initializer/zeros*
validate_shape(*
_output_shapes
:@*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_1/biases*
use_locking(*
T0
?
*ssd_300_vgg/resblock0_1/conv_1/biases/readIdentity%ssd_300_vgg/resblock0_1/conv_1/biases*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_1/biases*
T0*
_output_shapes
:@
}
,ssd_300_vgg/resblock0_1/conv_1/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
%ssd_300_vgg/resblock0_1/conv_1/Conv2DConv2Dssd_300_vgg/resblock0_1/Relu_1+ssd_300_vgg/resblock0_1/conv_1/weights/read*
paddingSAME*&
_output_shapes
:88@*
	dilations
*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0
?
&ssd_300_vgg/resblock0_1/conv_1/BiasAddBiasAdd%ssd_300_vgg/resblock0_1/conv_1/Conv2D*ssd_300_vgg/resblock0_1/conv_1/biases/read*
data_formatNHWC*
T0*&
_output_shapes
:88@
?
ssd_300_vgg/resblock0_1/addAdd&ssd_300_vgg/resblock0_1/conv_1/BiasAddssd_300_vgg/resblock0_0/add*&
_output_shapes
:88@*
T0
?
;ssd_300_vgg/resblock1_0/batch_norm_0/beta/Initializer/zerosConst*
valueB@*    *<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_0/beta*
_output_shapes
:@*
dtype0
?
)ssd_300_vgg/resblock1_0/batch_norm_0/beta
VariableV2*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_0/beta*
shared_name *
	container *
shape:@*
_output_shapes
:@*
dtype0
?
0ssd_300_vgg/resblock1_0/batch_norm_0/beta/AssignAssign)ssd_300_vgg/resblock1_0/batch_norm_0/beta;ssd_300_vgg/resblock1_0/batch_norm_0/beta/Initializer/zeros*
validate_shape(*
T0*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_0/beta*
_output_shapes
:@
?
.ssd_300_vgg/resblock1_0/batch_norm_0/beta/readIdentity)ssd_300_vgg/resblock1_0/batch_norm_0/beta*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_0/beta*
_output_shapes
:@*
T0
?
;ssd_300_vgg/resblock1_0/batch_norm_0/gamma/Initializer/onesConst*
_output_shapes
:@*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_0/gamma*
dtype0*
valueB@*  ??
?
*ssd_300_vgg/resblock1_0/batch_norm_0/gamma
VariableV2*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_0/gamma*
_output_shapes
:@*
	container *
dtype0*
shape:@*
shared_name 
?
1ssd_300_vgg/resblock1_0/batch_norm_0/gamma/AssignAssign*ssd_300_vgg/resblock1_0/batch_norm_0/gamma;ssd_300_vgg/resblock1_0/batch_norm_0/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:@*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_0/gamma*
use_locking(
?
/ssd_300_vgg/resblock1_0/batch_norm_0/gamma/readIdentity*ssd_300_vgg/resblock1_0/batch_norm_0/gamma*
_output_shapes
:@*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_0/gamma*
T0
?
Bssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean*
valueB@*    
?
0ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean*
shape:@*
	container 
?
7ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/AssignAssign0ssd_300_vgg/resblock1_0/batch_norm_0/moving_meanBssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean
?
5ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/readIdentity0ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean*
T0
?
Essd_300_vgg/resblock1_0/batch_norm_0/moving_variance/Initializer/onesConst*
_output_shapes
:@*
dtype0*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance*
valueB@*  ??
?
4ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance
VariableV2*
dtype0*
shared_name *
shape:@*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance*
_output_shapes
:@*
	container 
?
;ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/AssignAssign4ssd_300_vgg/resblock1_0/batch_norm_0/moving_varianceEssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/Initializer/ones*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
?
9ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/readIdentity4ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance*
_output_shapes
:@
?
3ssd_300_vgg/resblock1_0/batch_norm_0/FusedBatchNormFusedBatchNormssd_300_vgg/resblock0_1/add/ssd_300_vgg/resblock1_0/batch_norm_0/gamma/read.ssd_300_vgg/resblock1_0/batch_norm_0/beta/read5ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/read9ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/read*
is_training( *
data_formatNHWC*
T0*
epsilon%??'7*>
_output_shapes,
*:88@:@:@:@:@
?
ssd_300_vgg/resblock1_0/ReluRelu3ssd_300_vgg/resblock1_0/batch_norm_0/FusedBatchNorm*&
_output_shapes
:88@*
T0
?
$ssd_300_vgg/resblock1_0/Pad/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
?
ssd_300_vgg/resblock1_0/PadPadssd_300_vgg/resblock1_0/Relu$ssd_300_vgg/resblock1_0/Pad/paddings*
T0*&
_output_shapes
:::@*
	Tpaddings0
?
Gssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      @   ?   *9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*
_output_shapes
:
?
Essd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/minConst*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*
_output_shapes
: *
valueB
 *?[q?*
dtype0
?
Essd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/maxConst*
valueB
 *?[q=*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*
_output_shapes
: 
?
Ossd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/shape*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*
seed2 *

seed *
dtype0*'
_output_shapes
:@?*
T0
?
Essd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*
_output_shapes
: *
T0
?
Essd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/sub*'
_output_shapes
:@?*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights
?
Assd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniformAddEssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*'
_output_shapes
:@?*
T0
?
&ssd_300_vgg/resblock1_0/conv_0/weights
VariableV2*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*
shape:@?*
	container *
shared_name *
dtype0*'
_output_shapes
:@?
?
-ssd_300_vgg/resblock1_0/conv_0/weights/AssignAssign&ssd_300_vgg/resblock1_0/conv_0/weightsAssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*
use_locking(*
T0*'
_output_shapes
:@?
?
+ssd_300_vgg/resblock1_0/conv_0/weights/readIdentity&ssd_300_vgg/resblock1_0/conv_0/weights*'
_output_shapes
:@?*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*
T0
?
Fssd_300_vgg/resblock1_0/conv_0/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:
?
Gssd_300_vgg/resblock1_0/conv_0/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock1_0/conv_0/weights/read*
T0*
_output_shapes
: 
?
@ssd_300_vgg/resblock1_0/conv_0/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock1_0/conv_0/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock1_0/conv_0/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
7ssd_300_vgg/resblock1_0/conv_0/biases/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_0/biases*
dtype0
?
%ssd_300_vgg/resblock1_0/conv_0/biases
VariableV2*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_0/biases*
shared_name *
dtype0*
_output_shapes	
:?*
	container *
shape:?
?
,ssd_300_vgg/resblock1_0/conv_0/biases/AssignAssign%ssd_300_vgg/resblock1_0/conv_0/biases7ssd_300_vgg/resblock1_0/conv_0/biases/Initializer/zeros*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_0/biases
?
*ssd_300_vgg/resblock1_0/conv_0/biases/readIdentity%ssd_300_vgg/resblock1_0/conv_0/biases*
_output_shapes	
:?*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_0/biases
}
,ssd_300_vgg/resblock1_0/conv_0/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
%ssd_300_vgg/resblock1_0/conv_0/Conv2DConv2Dssd_300_vgg/resblock1_0/Pad+ssd_300_vgg/resblock1_0/conv_0/weights/read*
data_formatNHWC*'
_output_shapes
:?*
strides
*
paddingVALID*
use_cudnn_on_gpu(*
	dilations
*
T0
?
&ssd_300_vgg/resblock1_0/conv_0/BiasAddBiasAdd%ssd_300_vgg/resblock1_0/conv_0/Conv2D*ssd_300_vgg/resblock1_0/conv_0/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:?
?
&ssd_300_vgg/resblock1_0/Pad_1/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 
?
ssd_300_vgg/resblock1_0/Pad_1Padssd_300_vgg/resblock0_1/add&ssd_300_vgg/resblock1_0/Pad_1/paddings*
T0*&
_output_shapes
:88@*
	Tpaddings0
?
Jssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      @   ?   *
_output_shapes
:*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights
?
Hssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/minConst*
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights*
_output_shapes
: *
valueB
 *?5?
?
Hssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *?5>*
_output_shapes
: *<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights
?
Rssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/RandomUniformRandomUniformJssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/shape*

seed *
T0*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights*
dtype0*
seed2 *'
_output_shapes
:@?
?
Hssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/subSubHssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/maxHssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights
?
Hssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/mulMulRssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/RandomUniformHssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/sub*'
_output_shapes
:@?*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights
?
Dssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniformAddHssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/mulHssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform/min*'
_output_shapes
:@?*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights*
T0
?
)ssd_300_vgg/resblock1_0/conv_init/weights
VariableV2*
shared_name *'
_output_shapes
:@?*
	container *
dtype0*
shape:@?*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights
?
0ssd_300_vgg/resblock1_0/conv_init/weights/AssignAssign)ssd_300_vgg/resblock1_0/conv_init/weightsDssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform*'
_output_shapes
:@?*
use_locking(*
T0*
validate_shape(*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights
?
.ssd_300_vgg/resblock1_0/conv_init/weights/readIdentity)ssd_300_vgg/resblock1_0/conv_init/weights*'
_output_shapes
:@?*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights*
T0
?
Issd_300_vgg/resblock1_0/conv_init/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
_output_shapes
: *
dtype0
?
Jssd_300_vgg/resblock1_0/conv_init/kernel/Regularizer/l2_regularizer/L2LossL2Loss.ssd_300_vgg/resblock1_0/conv_init/weights/read*
T0*
_output_shapes
: 
?
Cssd_300_vgg/resblock1_0/conv_init/kernel/Regularizer/l2_regularizerMulIssd_300_vgg/resblock1_0/conv_init/kernel/Regularizer/l2_regularizer/scaleJssd_300_vgg/resblock1_0/conv_init/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
:ssd_300_vgg/resblock1_0/conv_init/biases/Initializer/zerosConst*;
_class1
/-loc:@ssd_300_vgg/resblock1_0/conv_init/biases*
dtype0*
_output_shapes	
:?*
valueB?*    
?
(ssd_300_vgg/resblock1_0/conv_init/biases
VariableV2*
shared_name *
_output_shapes	
:?*
dtype0*
	container *
shape:?*;
_class1
/-loc:@ssd_300_vgg/resblock1_0/conv_init/biases
?
/ssd_300_vgg/resblock1_0/conv_init/biases/AssignAssign(ssd_300_vgg/resblock1_0/conv_init/biases:ssd_300_vgg/resblock1_0/conv_init/biases/Initializer/zeros*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(*;
_class1
/-loc:@ssd_300_vgg/resblock1_0/conv_init/biases
?
-ssd_300_vgg/resblock1_0/conv_init/biases/readIdentity(ssd_300_vgg/resblock1_0/conv_init/biases*
T0*
_output_shapes	
:?*;
_class1
/-loc:@ssd_300_vgg/resblock1_0/conv_init/biases
?
/ssd_300_vgg/resblock1_0/conv_init/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
(ssd_300_vgg/resblock1_0/conv_init/Conv2DConv2Dssd_300_vgg/resblock1_0/Pad_1.ssd_300_vgg/resblock1_0/conv_init/weights/read*
paddingVALID*
	dilations
*
data_formatNHWC*'
_output_shapes
:?*
strides
*
T0*
use_cudnn_on_gpu(
?
)ssd_300_vgg/resblock1_0/conv_init/BiasAddBiasAdd(ssd_300_vgg/resblock1_0/conv_init/Conv2D-ssd_300_vgg/resblock1_0/conv_init/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:?
?
;ssd_300_vgg/resblock1_0/batch_norm_1/beta/Initializer/zerosConst*
valueB?*    *
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_1/beta*
_output_shapes	
:?
?
)ssd_300_vgg/resblock1_0/batch_norm_1/beta
VariableV2*
dtype0*
shape:?*
shared_name *
_output_shapes	
:?*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_1/beta*
	container 
?
0ssd_300_vgg/resblock1_0/batch_norm_1/beta/AssignAssign)ssd_300_vgg/resblock1_0/batch_norm_1/beta;ssd_300_vgg/resblock1_0/batch_norm_1/beta/Initializer/zeros*
_output_shapes	
:?*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_1/beta*
validate_shape(*
T0*
use_locking(
?
.ssd_300_vgg/resblock1_0/batch_norm_1/beta/readIdentity)ssd_300_vgg/resblock1_0/batch_norm_1/beta*
_output_shapes	
:?*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_1/beta
?
;ssd_300_vgg/resblock1_0/batch_norm_1/gamma/Initializer/onesConst*
dtype0*
valueB?*  ??*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_1/gamma
?
*ssd_300_vgg/resblock1_0/batch_norm_1/gamma
VariableV2*
shape:?*
	container *=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_1/gamma*
shared_name *
dtype0*
_output_shapes	
:?
?
1ssd_300_vgg/resblock1_0/batch_norm_1/gamma/AssignAssign*ssd_300_vgg/resblock1_0/batch_norm_1/gamma;ssd_300_vgg/resblock1_0/batch_norm_1/gamma/Initializer/ones*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_1/gamma*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
/ssd_300_vgg/resblock1_0/batch_norm_1/gamma/readIdentity*ssd_300_vgg/resblock1_0/batch_norm_1/gamma*
_output_shapes	
:?*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_1/gamma
?
Bssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/Initializer/zerosConst*
valueB?*    *C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean*
dtype0*
_output_shapes	
:?
?
0ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean
VariableV2*
dtype0*
shared_name *C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean*
shape:?*
_output_shapes	
:?*
	container 
?
7ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/AssignAssign0ssd_300_vgg/resblock1_0/batch_norm_1/moving_meanBssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/Initializer/zeros*
validate_shape(*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean*
use_locking(*
T0*
_output_shapes	
:?
?
5ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/readIdentity0ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean*
_output_shapes	
:?*
T0*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean
?
Essd_300_vgg/resblock1_0/batch_norm_1/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance
?
4ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance
VariableV2*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance*
	container *
shared_name *
shape:?*
dtype0
?
;ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/AssignAssign4ssd_300_vgg/resblock1_0/batch_norm_1/moving_varianceEssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/Initializer/ones*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?
?
9ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/readIdentity4ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance*
T0*
_output_shapes	
:?
?
3ssd_300_vgg/resblock1_0/batch_norm_1/FusedBatchNormFusedBatchNorm&ssd_300_vgg/resblock1_0/conv_0/BiasAdd/ssd_300_vgg/resblock1_0/batch_norm_1/gamma/read.ssd_300_vgg/resblock1_0/batch_norm_1/beta/read5ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/read9ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/read*
T0*
is_training( *
data_formatNHWC*C
_output_shapes1
/:?:?:?:?:?*
epsilon%??'7
?
ssd_300_vgg/resblock1_0/Relu_1Relu3ssd_300_vgg/resblock1_0/batch_norm_1/FusedBatchNorm*
T0*'
_output_shapes
:?
?
Gssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/shapeConst*%
valueB"      ?   ?   *
dtype0*
_output_shapes
:*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights
?
Essd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/minConst*
valueB
 *?Q?*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights*
dtype0*
_output_shapes
: 
?
Essd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/maxConst*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights*
valueB
 *?Q=*
_output_shapes
: *
dtype0
?
Ossd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/shape*
seed2 *(
_output_shapes
:??*

seed *
T0*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights
?
Essd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights*
T0*
_output_shapes
: 
?
Essd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights*(
_output_shapes
:??*
T0
?
Assd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniformAddEssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform/min*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights*
T0
?
&ssd_300_vgg/resblock1_0/conv_1/weights
VariableV2*(
_output_shapes
:??*
shape:??*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights*
shared_name *
	container 
?
-ssd_300_vgg/resblock1_0/conv_1/weights/AssignAssign&ssd_300_vgg/resblock1_0/conv_1/weightsAssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights*
use_locking(*
T0*(
_output_shapes
:??
?
+ssd_300_vgg/resblock1_0/conv_1/weights/readIdentity&ssd_300_vgg/resblock1_0/conv_1/weights*
T0*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights
?
Fssd_300_vgg/resblock1_0/conv_1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Gssd_300_vgg/resblock1_0/conv_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock1_0/conv_1/weights/read*
_output_shapes
: *
T0
?
@ssd_300_vgg/resblock1_0/conv_1/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock1_0/conv_1/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock1_0/conv_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
7ssd_300_vgg/resblock1_0/conv_1/biases/Initializer/zerosConst*
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_1/biases*
dtype0*
valueB?*    
?
%ssd_300_vgg/resblock1_0/conv_1/biases
VariableV2*
_output_shapes	
:?*
	container *
dtype0*
shape:?*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_1/biases*
shared_name 
?
,ssd_300_vgg/resblock1_0/conv_1/biases/AssignAssign%ssd_300_vgg/resblock1_0/conv_1/biases7ssd_300_vgg/resblock1_0/conv_1/biases/Initializer/zeros*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_1/biases
?
*ssd_300_vgg/resblock1_0/conv_1/biases/readIdentity%ssd_300_vgg/resblock1_0/conv_1/biases*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_1/biases*
_output_shapes	
:?*
T0
}
,ssd_300_vgg/resblock1_0/conv_1/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
%ssd_300_vgg/resblock1_0/conv_1/Conv2DConv2Dssd_300_vgg/resblock1_0/Relu_1+ssd_300_vgg/resblock1_0/conv_1/weights/read*
	dilations
*
T0*'
_output_shapes
:?*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*
strides

?
&ssd_300_vgg/resblock1_0/conv_1/BiasAddBiasAdd%ssd_300_vgg/resblock1_0/conv_1/Conv2D*ssd_300_vgg/resblock1_0/conv_1/biases/read*
T0*'
_output_shapes
:?*
data_formatNHWC
?
ssd_300_vgg/resblock1_0/addAdd&ssd_300_vgg/resblock1_0/conv_1/BiasAdd)ssd_300_vgg/resblock1_0/conv_init/BiasAdd*
T0*'
_output_shapes
:?
?
;ssd_300_vgg/resblock1_1/batch_norm_0/beta/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_0/beta
?
)ssd_300_vgg/resblock1_1/batch_norm_0/beta
VariableV2*
shape:?*
shared_name *
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_0/beta*
	container *
_output_shapes	
:?
?
0ssd_300_vgg/resblock1_1/batch_norm_0/beta/AssignAssign)ssd_300_vgg/resblock1_1/batch_norm_0/beta;ssd_300_vgg/resblock1_1/batch_norm_0/beta/Initializer/zeros*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_0/beta*
_output_shapes	
:?*
T0*
validate_shape(
?
.ssd_300_vgg/resblock1_1/batch_norm_0/beta/readIdentity)ssd_300_vgg/resblock1_1/batch_norm_0/beta*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_0/beta*
_output_shapes	
:?
?
;ssd_300_vgg/resblock1_1/batch_norm_0/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_0/gamma
?
*ssd_300_vgg/resblock1_1/batch_norm_0/gamma
VariableV2*
	container *
shared_name *
dtype0*
shape:?*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_0/gamma*
_output_shapes	
:?
?
1ssd_300_vgg/resblock1_1/batch_norm_0/gamma/AssignAssign*ssd_300_vgg/resblock1_1/batch_norm_0/gamma;ssd_300_vgg/resblock1_1/batch_norm_0/gamma/Initializer/ones*
validate_shape(*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_0/gamma*
_output_shapes	
:?*
use_locking(
?
/ssd_300_vgg/resblock1_1/batch_norm_0/gamma/readIdentity*ssd_300_vgg/resblock1_1/batch_norm_0/gamma*
T0*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_0/gamma
?
Bssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean
?
0ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean
VariableV2*
dtype0*
	container *
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean*
shared_name *
shape:?
?
7ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/AssignAssign0ssd_300_vgg/resblock1_1/batch_norm_0/moving_meanBssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/Initializer/zeros*
T0*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean*
use_locking(*
validate_shape(
?
5ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/readIdentity0ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean*
_output_shapes	
:?*
T0
?
Essd_300_vgg/resblock1_1/batch_norm_0/moving_variance/Initializer/onesConst*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance*
dtype0*
valueB?*  ??
?
4ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance
VariableV2*
_output_shapes	
:?*
shape:?*
dtype0*
	container *G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance*
shared_name 
?
;ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/AssignAssign4ssd_300_vgg/resblock1_1/batch_norm_0/moving_varianceEssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/Initializer/ones*
use_locking(*
T0*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance*
validate_shape(
?
9ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/readIdentity4ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance*
T0*
_output_shapes	
:?
?
3ssd_300_vgg/resblock1_1/batch_norm_0/FusedBatchNormFusedBatchNormssd_300_vgg/resblock1_0/add/ssd_300_vgg/resblock1_1/batch_norm_0/gamma/read.ssd_300_vgg/resblock1_1/batch_norm_0/beta/read5ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/read9ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/read*
is_training( *
data_formatNHWC*
epsilon%??'7*C
_output_shapes1
/:?:?:?:?:?*
T0
?
ssd_300_vgg/resblock1_1/ReluRelu3ssd_300_vgg/resblock1_1/batch_norm_0/FusedBatchNorm*'
_output_shapes
:?*
T0
?
Gssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights*
dtype0*%
valueB"      ?   ?   
?
Essd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *?Q?*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights
?
Essd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *?Q=*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights
?
Ossd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/shape*
seed2 *9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights*(
_output_shapes
:??*
T0*

seed *
dtype0
?
Essd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/min*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights*
T0
?
Essd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights*(
_output_shapes
:??*
T0
?
Assd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniformAddEssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform/min*(
_output_shapes
:??*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights
?
&ssd_300_vgg/resblock1_1/conv_0/weights
VariableV2*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights*(
_output_shapes
:??*
	container *
shape:??*
shared_name *
dtype0
?
-ssd_300_vgg/resblock1_1/conv_0/weights/AssignAssign&ssd_300_vgg/resblock1_1/conv_0/weightsAssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights*
validate_shape(*
T0*(
_output_shapes
:??*
use_locking(
?
+ssd_300_vgg/resblock1_1/conv_0/weights/readIdentity&ssd_300_vgg/resblock1_1/conv_0/weights*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights*
T0
?
Fssd_300_vgg/resblock1_1/conv_0/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:
?
Gssd_300_vgg/resblock1_1/conv_0/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock1_1/conv_0/weights/read*
_output_shapes
: *
T0
?
@ssd_300_vgg/resblock1_1/conv_0/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock1_1/conv_0/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock1_1/conv_0/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
7ssd_300_vgg/resblock1_1/conv_0/biases/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_0/biases
?
%ssd_300_vgg/resblock1_1/conv_0/biases
VariableV2*
shared_name *8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_0/biases*
shape:?*
	container *
dtype0*
_output_shapes	
:?
?
,ssd_300_vgg/resblock1_1/conv_0/biases/AssignAssign%ssd_300_vgg/resblock1_1/conv_0/biases7ssd_300_vgg/resblock1_1/conv_0/biases/Initializer/zeros*
use_locking(*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_0/biases*
_output_shapes	
:?*
T0
?
*ssd_300_vgg/resblock1_1/conv_0/biases/readIdentity%ssd_300_vgg/resblock1_1/conv_0/biases*
_output_shapes	
:?*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_0/biases
}
,ssd_300_vgg/resblock1_1/conv_0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
%ssd_300_vgg/resblock1_1/conv_0/Conv2DConv2Dssd_300_vgg/resblock1_1/Relu+ssd_300_vgg/resblock1_1/conv_0/weights/read*'
_output_shapes
:?*
data_formatNHWC*
	dilations
*
T0*
use_cudnn_on_gpu(*
paddingSAME*
strides

?
&ssd_300_vgg/resblock1_1/conv_0/BiasAddBiasAdd%ssd_300_vgg/resblock1_1/conv_0/Conv2D*ssd_300_vgg/resblock1_1/conv_0/biases/read*
T0*'
_output_shapes
:?*
data_formatNHWC
?
;ssd_300_vgg/resblock1_1/batch_norm_1/beta/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_1/beta
?
)ssd_300_vgg/resblock1_1/batch_norm_1/beta
VariableV2*
shape:?*
_output_shapes	
:?*
shared_name *
dtype0*
	container *<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_1/beta
?
0ssd_300_vgg/resblock1_1/batch_norm_1/beta/AssignAssign)ssd_300_vgg/resblock1_1/batch_norm_1/beta;ssd_300_vgg/resblock1_1/batch_norm_1/beta/Initializer/zeros*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_1/beta*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
.ssd_300_vgg/resblock1_1/batch_norm_1/beta/readIdentity)ssd_300_vgg/resblock1_1/batch_norm_1/beta*
T0*
_output_shapes	
:?*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_1/beta
?
;ssd_300_vgg/resblock1_1/batch_norm_1/gamma/Initializer/onesConst*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_1/gamma*
dtype0*
valueB?*  ??*
_output_shapes	
:?
?
*ssd_300_vgg/resblock1_1/batch_norm_1/gamma
VariableV2*
	container *
shared_name *
dtype0*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_1/gamma*
_output_shapes	
:?*
shape:?
?
1ssd_300_vgg/resblock1_1/batch_norm_1/gamma/AssignAssign*ssd_300_vgg/resblock1_1/batch_norm_1/gamma;ssd_300_vgg/resblock1_1/batch_norm_1/gamma/Initializer/ones*
_output_shapes	
:?*
validate_shape(*
use_locking(*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_1/gamma*
T0
?
/ssd_300_vgg/resblock1_1/batch_norm_1/gamma/readIdentity*ssd_300_vgg/resblock1_1/batch_norm_1/gamma*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_1/gamma*
T0
?
Bssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean
?
0ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean
VariableV2*
_output_shapes	
:?*
shape:?*
shared_name *
dtype0*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean*
	container 
?
7ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/AssignAssign0ssd_300_vgg/resblock1_1/batch_norm_1/moving_meanBssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/Initializer/zeros*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean*
_output_shapes	
:?*
T0*
use_locking(*
validate_shape(
?
5ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/readIdentity0ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean*
T0*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean
?
Essd_300_vgg/resblock1_1/batch_norm_1/moving_variance/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*
dtype0*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance
?
4ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
shared_name *G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance*
_output_shapes	
:?
?
;ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/AssignAssign4ssd_300_vgg/resblock1_1/batch_norm_1/moving_varianceEssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/Initializer/ones*
use_locking(*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance*
_output_shapes	
:?*
validate_shape(
?
9ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/readIdentity4ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance*
T0
?
3ssd_300_vgg/resblock1_1/batch_norm_1/FusedBatchNormFusedBatchNorm&ssd_300_vgg/resblock1_1/conv_0/BiasAdd/ssd_300_vgg/resblock1_1/batch_norm_1/gamma/read.ssd_300_vgg/resblock1_1/batch_norm_1/beta/read5ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/read9ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/read*C
_output_shapes1
/:?:?:?:?:?*
is_training( *
T0*
data_formatNHWC*
epsilon%??'7
?
ssd_300_vgg/resblock1_1/Relu_1Relu3ssd_300_vgg/resblock1_1/batch_norm_1/FusedBatchNorm*
T0*'
_output_shapes
:?
?
Gssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"      ?   ?   *9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights
?
Essd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *?Q?*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights
?
Essd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights*
valueB
 *?Q=
?
Ossd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/shape*
dtype0*(
_output_shapes
:??*

seed *
seed2 *9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights*
T0
?
Essd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights*
T0*
_output_shapes
: 
?
Essd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights
?
Assd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniformAddEssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights*(
_output_shapes
:??
?
&ssd_300_vgg/resblock1_1/conv_1/weights
VariableV2*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights*
shape:??*
dtype0*(
_output_shapes
:??*
shared_name *
	container 
?
-ssd_300_vgg/resblock1_1/conv_1/weights/AssignAssign&ssd_300_vgg/resblock1_1/conv_1/weightsAssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform*
T0*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights*
use_locking(*
validate_shape(
?
+ssd_300_vgg/resblock1_1/conv_1/weights/readIdentity&ssd_300_vgg/resblock1_1/conv_1/weights*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights*(
_output_shapes
:??
?
Fssd_300_vgg/resblock1_1/conv_1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *o:*
dtype0
?
Gssd_300_vgg/resblock1_1/conv_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock1_1/conv_1/weights/read*
T0*
_output_shapes
: 
?
@ssd_300_vgg/resblock1_1/conv_1/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock1_1/conv_1/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock1_1/conv_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
7ssd_300_vgg/resblock1_1/conv_1/biases/Initializer/zerosConst*
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_1/biases*
dtype0*
valueB?*    
?
%ssd_300_vgg/resblock1_1/conv_1/biases
VariableV2*
shape:?*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_1/biases*
	container *
shared_name *
dtype0*
_output_shapes	
:?
?
,ssd_300_vgg/resblock1_1/conv_1/biases/AssignAssign%ssd_300_vgg/resblock1_1/conv_1/biases7ssd_300_vgg/resblock1_1/conv_1/biases/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_1/biases
?
*ssd_300_vgg/resblock1_1/conv_1/biases/readIdentity%ssd_300_vgg/resblock1_1/conv_1/biases*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_1/biases*
T0*
_output_shapes	
:?
}
,ssd_300_vgg/resblock1_1/conv_1/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
%ssd_300_vgg/resblock1_1/conv_1/Conv2DConv2Dssd_300_vgg/resblock1_1/Relu_1+ssd_300_vgg/resblock1_1/conv_1/weights/read*
data_formatNHWC*'
_output_shapes
:?*
T0*
use_cudnn_on_gpu(*
strides
*
	dilations
*
paddingSAME
?
&ssd_300_vgg/resblock1_1/conv_1/BiasAddBiasAdd%ssd_300_vgg/resblock1_1/conv_1/Conv2D*ssd_300_vgg/resblock1_1/conv_1/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:?
?
ssd_300_vgg/resblock1_1/addAdd&ssd_300_vgg/resblock1_1/conv_1/BiasAddssd_300_vgg/resblock1_0/add*
T0*'
_output_shapes
:?
?
;ssd_300_vgg/resblock2_0/batch_norm_0/beta/Initializer/zerosConst*
valueB?*    *<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_0/beta*
_output_shapes	
:?*
dtype0
?
)ssd_300_vgg/resblock2_0/batch_norm_0/beta
VariableV2*
_output_shapes	
:?*
	container *
dtype0*
shared_name *
shape:?*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_0/beta
?
0ssd_300_vgg/resblock2_0/batch_norm_0/beta/AssignAssign)ssd_300_vgg/resblock2_0/batch_norm_0/beta;ssd_300_vgg/resblock2_0/batch_norm_0/beta/Initializer/zeros*
T0*
_output_shapes	
:?*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_0/beta*
validate_shape(*
use_locking(
?
.ssd_300_vgg/resblock2_0/batch_norm_0/beta/readIdentity)ssd_300_vgg/resblock2_0/batch_norm_0/beta*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_0/beta*
T0*
_output_shapes	
:?
?
;ssd_300_vgg/resblock2_0/batch_norm_0/gamma/Initializer/onesConst*
dtype0*
valueB?*  ??*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_0/gamma*
_output_shapes	
:?
?
*ssd_300_vgg/resblock2_0/batch_norm_0/gamma
VariableV2*
dtype0*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_0/gamma*
	container *
_output_shapes	
:?*
shape:?*
shared_name 
?
1ssd_300_vgg/resblock2_0/batch_norm_0/gamma/AssignAssign*ssd_300_vgg/resblock2_0/batch_norm_0/gamma;ssd_300_vgg/resblock2_0/batch_norm_0/gamma/Initializer/ones*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_0/gamma
?
/ssd_300_vgg/resblock2_0/batch_norm_0/gamma/readIdentity*ssd_300_vgg/resblock2_0/batch_norm_0/gamma*
T0*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_0/gamma
?
Bssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean
?
0ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean
VariableV2*
dtype0*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean*
_output_shapes	
:?*
shape:?*
shared_name *
	container 
?
7ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/AssignAssign0ssd_300_vgg/resblock2_0/batch_norm_0/moving_meanBssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/Initializer/zeros*
use_locking(*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean*
T0*
validate_shape(*
_output_shapes	
:?
?
5ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/readIdentity0ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean*
T0*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean*
_output_shapes	
:?
?
Essd_300_vgg/resblock2_0/batch_norm_0/moving_variance/Initializer/onesConst*
dtype0*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance*
valueB?*  ??*
_output_shapes	
:?
?
4ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance
VariableV2*
shape:?*
shared_name *
	container *G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance*
_output_shapes	
:?*
dtype0
?
;ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/AssignAssign4ssd_300_vgg/resblock2_0/batch_norm_0/moving_varianceEssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/Initializer/ones*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance
?
9ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/readIdentity4ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance*
T0
?
3ssd_300_vgg/resblock2_0/batch_norm_0/FusedBatchNormFusedBatchNormssd_300_vgg/resblock1_1/add/ssd_300_vgg/resblock2_0/batch_norm_0/gamma/read.ssd_300_vgg/resblock2_0/batch_norm_0/beta/read5ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/read9ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/read*C
_output_shapes1
/:?:?:?:?:?*
is_training( *
epsilon%??'7*
T0*
data_formatNHWC
?
ssd_300_vgg/resblock2_0/ReluRelu3ssd_300_vgg/resblock2_0/batch_norm_0/FusedBatchNorm*
T0*'
_output_shapes
:?
?
$ssd_300_vgg/resblock2_0/Pad/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
?
ssd_300_vgg/resblock2_0/PadPadssd_300_vgg/resblock2_0/Relu$ssd_300_vgg/resblock2_0/Pad/paddings*'
_output_shapes
:?*
T0*
	Tpaddings0
?
Gssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      ?      *9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*
_output_shapes
:
?
Essd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*
valueB
 *??*?
?
Essd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/maxConst*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*
_output_shapes
: *
dtype0*
valueB
 *??*=
?
Ossd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/shape*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*
T0*

seed *
seed2 *(
_output_shapes
:??*
dtype0
?
Essd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights
?
Essd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*(
_output_shapes
:??*
T0
?
Assd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniformAddEssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*(
_output_shapes
:??*
T0
?
&ssd_300_vgg/resblock2_0/conv_0/weights
VariableV2*
	container *
shape:??*(
_output_shapes
:??*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*
shared_name 
?
-ssd_300_vgg/resblock2_0/conv_0/weights/AssignAssign&ssd_300_vgg/resblock2_0/conv_0/weightsAssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*(
_output_shapes
:??*
T0*
use_locking(
?
+ssd_300_vgg/resblock2_0/conv_0/weights/readIdentity&ssd_300_vgg/resblock2_0/conv_0/weights*
T0*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights
?
Fssd_300_vgg/resblock2_0/conv_0/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
?
Gssd_300_vgg/resblock2_0/conv_0/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock2_0/conv_0/weights/read*
T0*
_output_shapes
: 
?
@ssd_300_vgg/resblock2_0/conv_0/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock2_0/conv_0/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock2_0/conv_0/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
7ssd_300_vgg/resblock2_0/conv_0/biases/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*
valueB?*    *8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_0/biases
?
%ssd_300_vgg/resblock2_0/conv_0/biases
VariableV2*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_0/biases
?
,ssd_300_vgg/resblock2_0/conv_0/biases/AssignAssign%ssd_300_vgg/resblock2_0/conv_0/biases7ssd_300_vgg/resblock2_0/conv_0/biases/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_0/biases
?
*ssd_300_vgg/resblock2_0/conv_0/biases/readIdentity%ssd_300_vgg/resblock2_0/conv_0/biases*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_0/biases*
_output_shapes	
:?
}
,ssd_300_vgg/resblock2_0/conv_0/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
%ssd_300_vgg/resblock2_0/conv_0/Conv2DConv2Dssd_300_vgg/resblock2_0/Pad+ssd_300_vgg/resblock2_0/conv_0/weights/read*
T0*
	dilations
*'
_output_shapes
:?*
strides
*
paddingVALID*
use_cudnn_on_gpu(*
data_formatNHWC
?
&ssd_300_vgg/resblock2_0/conv_0/BiasAddBiasAdd%ssd_300_vgg/resblock2_0/conv_0/Conv2D*ssd_300_vgg/resblock2_0/conv_0/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:?
?
&ssd_300_vgg/resblock2_0/Pad_1/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                                 
?
ssd_300_vgg/resblock2_0/Pad_1Padssd_300_vgg/resblock1_1/add&ssd_300_vgg/resblock2_0/Pad_1/paddings*
	Tpaddings0*'
_output_shapes
:?*
T0
?
Jssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/shapeConst*%
valueB"      ?      *<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*
_output_shapes
:*
dtype0
?
Hssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/minConst*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*
dtype0*
_output_shapes
: *
valueB
 *   ?
?
Hssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *   >*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*
dtype0
?
Rssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/RandomUniformRandomUniformJssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/shape*(
_output_shapes
:??*

seed *
seed2 *
T0*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*
dtype0
?
Hssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/subSubHssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/maxHssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/min*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*
_output_shapes
: 
?
Hssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/mulMulRssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/RandomUniformHssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights
?
Dssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniformAddHssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/mulHssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform/min*(
_output_shapes
:??*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights
?
)ssd_300_vgg/resblock2_0/conv_init/weights
VariableV2*
	container *
shared_name *
dtype0*
shape:??*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*(
_output_shapes
:??
?
0ssd_300_vgg/resblock2_0/conv_init/weights/AssignAssign)ssd_300_vgg/resblock2_0/conv_init/weightsDssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*
validate_shape(*
use_locking(*
T0*(
_output_shapes
:??
?
.ssd_300_vgg/resblock2_0/conv_init/weights/readIdentity)ssd_300_vgg/resblock2_0/conv_init/weights*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*(
_output_shapes
:??*
T0
?
Issd_300_vgg/resblock2_0/conv_init/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:
?
Jssd_300_vgg/resblock2_0/conv_init/kernel/Regularizer/l2_regularizer/L2LossL2Loss.ssd_300_vgg/resblock2_0/conv_init/weights/read*
_output_shapes
: *
T0
?
Cssd_300_vgg/resblock2_0/conv_init/kernel/Regularizer/l2_regularizerMulIssd_300_vgg/resblock2_0/conv_init/kernel/Regularizer/l2_regularizer/scaleJssd_300_vgg/resblock2_0/conv_init/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
:ssd_300_vgg/resblock2_0/conv_init/biases/Initializer/zerosConst*
valueB?*    *;
_class1
/-loc:@ssd_300_vgg/resblock2_0/conv_init/biases*
dtype0*
_output_shapes	
:?
?
(ssd_300_vgg/resblock2_0/conv_init/biases
VariableV2*
shared_name *
dtype0*
_output_shapes	
:?*
shape:?*
	container *;
_class1
/-loc:@ssd_300_vgg/resblock2_0/conv_init/biases
?
/ssd_300_vgg/resblock2_0/conv_init/biases/AssignAssign(ssd_300_vgg/resblock2_0/conv_init/biases:ssd_300_vgg/resblock2_0/conv_init/biases/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*;
_class1
/-loc:@ssd_300_vgg/resblock2_0/conv_init/biases*
validate_shape(
?
-ssd_300_vgg/resblock2_0/conv_init/biases/readIdentity(ssd_300_vgg/resblock2_0/conv_init/biases*
_output_shapes	
:?*;
_class1
/-loc:@ssd_300_vgg/resblock2_0/conv_init/biases*
T0
?
/ssd_300_vgg/resblock2_0/conv_init/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
(ssd_300_vgg/resblock2_0/conv_init/Conv2DConv2Dssd_300_vgg/resblock2_0/Pad_1.ssd_300_vgg/resblock2_0/conv_init/weights/read*
T0*
data_formatNHWC*
use_cudnn_on_gpu(*'
_output_shapes
:?*
	dilations
*
strides
*
paddingVALID
?
)ssd_300_vgg/resblock2_0/conv_init/BiasAddBiasAdd(ssd_300_vgg/resblock2_0/conv_init/Conv2D-ssd_300_vgg/resblock2_0/conv_init/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:?
?
;ssd_300_vgg/resblock2_0/batch_norm_1/beta/Initializer/zerosConst*
dtype0*
valueB?*    *<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_1/beta*
_output_shapes	
:?
?
)ssd_300_vgg/resblock2_0/batch_norm_1/beta
VariableV2*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_1/beta*
dtype0*
_output_shapes	
:?*
	container *
shared_name *
shape:?
?
0ssd_300_vgg/resblock2_0/batch_norm_1/beta/AssignAssign)ssd_300_vgg/resblock2_0/batch_norm_1/beta;ssd_300_vgg/resblock2_0/batch_norm_1/beta/Initializer/zeros*
validate_shape(*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_1/beta*
use_locking(*
T0*
_output_shapes	
:?
?
.ssd_300_vgg/resblock2_0/batch_norm_1/beta/readIdentity)ssd_300_vgg/resblock2_0/batch_norm_1/beta*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_1/beta*
T0*
_output_shapes	
:?
?
;ssd_300_vgg/resblock2_0/batch_norm_1/gamma/Initializer/onesConst*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_1/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
*ssd_300_vgg/resblock2_0/batch_norm_1/gamma
VariableV2*
_output_shapes	
:?*
	container *
shared_name *
dtype0*
shape:?*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_1/gamma
?
1ssd_300_vgg/resblock2_0/batch_norm_1/gamma/AssignAssign*ssd_300_vgg/resblock2_0/batch_norm_1/gamma;ssd_300_vgg/resblock2_0/batch_norm_1/gamma/Initializer/ones*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_1/gamma
?
/ssd_300_vgg/resblock2_0/batch_norm_1/gamma/readIdentity*ssd_300_vgg/resblock2_0/batch_norm_1/gamma*
T0*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_1/gamma
?
Bssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean
?
0ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean
VariableV2*
shape:?*
shared_name *
dtype0*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean*
	container *
_output_shapes	
:?
?
7ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/AssignAssign0ssd_300_vgg/resblock2_0/batch_norm_1/moving_meanBssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean
?
5ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/readIdentity0ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean*
_output_shapes	
:?*
T0
?
Essd_300_vgg/resblock2_0/batch_norm_1/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance
?
4ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance
VariableV2*
dtype0*
shared_name *
	container *G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance*
_output_shapes	
:?*
shape:?
?
;ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/AssignAssign4ssd_300_vgg/resblock2_0/batch_norm_1/moving_varianceEssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/Initializer/ones*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance
?
9ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/readIdentity4ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance*
T0*
_output_shapes	
:?
?
3ssd_300_vgg/resblock2_0/batch_norm_1/FusedBatchNormFusedBatchNorm&ssd_300_vgg/resblock2_0/conv_0/BiasAdd/ssd_300_vgg/resblock2_0/batch_norm_1/gamma/read.ssd_300_vgg/resblock2_0/batch_norm_1/beta/read5ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/read9ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/read*
T0*C
_output_shapes1
/:?:?:?:?:?*
epsilon%??'7*
data_formatNHWC*
is_training( 
?
ssd_300_vgg/resblock2_0/Relu_1Relu3ssd_300_vgg/resblock2_0/batch_norm_1/FusedBatchNorm*
T0*'
_output_shapes
:?
?
Gssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/shapeConst*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*
dtype0*
_output_shapes
:*%
valueB"            
?
Essd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/minConst*
valueB
 *:??*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*
_output_shapes
: 
?
Essd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/maxConst*
valueB
 *:?=*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*
dtype0
?
Ossd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/shape*
dtype0*
T0*(
_output_shapes
:??*
seed2 *9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*

seed 
?
Essd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*
_output_shapes
: *
T0
?
Essd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*
T0
?
Assd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniformAddEssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights
?
&ssd_300_vgg/resblock2_0/conv_1/weights
VariableV2*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*
shape:??*
dtype0*
shared_name *(
_output_shapes
:??*
	container 
?
-ssd_300_vgg/resblock2_0/conv_1/weights/AssignAssign&ssd_300_vgg/resblock2_0/conv_1/weightsAssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*
T0*
use_locking(*
validate_shape(
?
+ssd_300_vgg/resblock2_0/conv_1/weights/readIdentity&ssd_300_vgg/resblock2_0/conv_1/weights*
T0*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights
?
Fssd_300_vgg/resblock2_0/conv_1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Gssd_300_vgg/resblock2_0/conv_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock2_0/conv_1/weights/read*
_output_shapes
: *
T0
?
@ssd_300_vgg/resblock2_0/conv_1/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock2_0/conv_1/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock2_0/conv_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
7ssd_300_vgg/resblock2_0/conv_1/biases/Initializer/zerosConst*
dtype0*
valueB?*    *8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_1/biases*
_output_shapes	
:?
?
%ssd_300_vgg/resblock2_0/conv_1/biases
VariableV2*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_1/biases*
_output_shapes	
:?*
	container *
shared_name *
shape:?*
dtype0
?
,ssd_300_vgg/resblock2_0/conv_1/biases/AssignAssign%ssd_300_vgg/resblock2_0/conv_1/biases7ssd_300_vgg/resblock2_0/conv_1/biases/Initializer/zeros*
T0*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_1/biases*
_output_shapes	
:?*
use_locking(
?
*ssd_300_vgg/resblock2_0/conv_1/biases/readIdentity%ssd_300_vgg/resblock2_0/conv_1/biases*
_output_shapes	
:?*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_1/biases
}
,ssd_300_vgg/resblock2_0/conv_1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
%ssd_300_vgg/resblock2_0/conv_1/Conv2DConv2Dssd_300_vgg/resblock2_0/Relu_1+ssd_300_vgg/resblock2_0/conv_1/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*'
_output_shapes
:?*
	dilations
*
paddingSAME
?
&ssd_300_vgg/resblock2_0/conv_1/BiasAddBiasAdd%ssd_300_vgg/resblock2_0/conv_1/Conv2D*ssd_300_vgg/resblock2_0/conv_1/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:?
?
ssd_300_vgg/resblock2_0/addAdd&ssd_300_vgg/resblock2_0/conv_1/BiasAdd)ssd_300_vgg/resblock2_0/conv_init/BiasAdd*
T0*'
_output_shapes
:?
?
;ssd_300_vgg/resblock2_1/batch_norm_0/beta/Initializer/zerosConst*
valueB?*    *<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_0/beta*
dtype0*
_output_shapes	
:?
?
)ssd_300_vgg/resblock2_1/batch_norm_0/beta
VariableV2*
	container *
shape:?*
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_0/beta*
shared_name *
_output_shapes	
:?
?
0ssd_300_vgg/resblock2_1/batch_norm_0/beta/AssignAssign)ssd_300_vgg/resblock2_1/batch_norm_0/beta;ssd_300_vgg/resblock2_1/batch_norm_0/beta/Initializer/zeros*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_0/beta*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(
?
.ssd_300_vgg/resblock2_1/batch_norm_0/beta/readIdentity)ssd_300_vgg/resblock2_1/batch_norm_0/beta*
T0*
_output_shapes	
:?*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_0/beta
?
;ssd_300_vgg/resblock2_1/batch_norm_0/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_0/gamma
?
*ssd_300_vgg/resblock2_1/batch_norm_0/gamma
VariableV2*
dtype0*
	container *
shared_name *=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_0/gamma*
_output_shapes	
:?*
shape:?
?
1ssd_300_vgg/resblock2_1/batch_norm_0/gamma/AssignAssign*ssd_300_vgg/resblock2_1/batch_norm_0/gamma;ssd_300_vgg/resblock2_1/batch_norm_0/gamma/Initializer/ones*
validate_shape(*
use_locking(*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_0/gamma*
T0*
_output_shapes	
:?
?
/ssd_300_vgg/resblock2_1/batch_norm_0/gamma/readIdentity*ssd_300_vgg/resblock2_1/batch_norm_0/gamma*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_0/gamma*
T0
?
Bssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean*
dtype0
?
0ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean
VariableV2*
_output_shapes	
:?*
dtype0*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean*
shape:?*
shared_name *
	container 
?
7ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/AssignAssign0ssd_300_vgg/resblock2_1/batch_norm_0/moving_meanBssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/Initializer/zeros*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean
?
5ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/readIdentity0ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean*
T0
?
Essd_300_vgg/resblock2_1/batch_norm_0/moving_variance/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*
dtype0*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance
?
4ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance
VariableV2*
dtype0*
	container *
_output_shapes	
:?*
shared_name *
shape:?*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance
?
;ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/AssignAssign4ssd_300_vgg/resblock2_1/batch_norm_0/moving_varianceEssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/Initializer/ones*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
9ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/readIdentity4ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance*
T0*
_output_shapes	
:?
?
3ssd_300_vgg/resblock2_1/batch_norm_0/FusedBatchNormFusedBatchNormssd_300_vgg/resblock2_0/add/ssd_300_vgg/resblock2_1/batch_norm_0/gamma/read.ssd_300_vgg/resblock2_1/batch_norm_0/beta/read5ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/read9ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/read*C
_output_shapes1
/:?:?:?:?:?*
epsilon%??'7*
T0*
is_training( *
data_formatNHWC
?
ssd_300_vgg/resblock2_1/ReluRelu3ssd_300_vgg/resblock2_1/batch_norm_0/FusedBatchNorm*
T0*'
_output_shapes
:?
?
Gssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights*
dtype0*%
valueB"            
?
Essd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *:??*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights
?
Essd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/maxConst*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights*
valueB
 *:?=*
dtype0*
_output_shapes
: 
?
Ossd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/shape*
dtype0*

seed *
seed2 *
T0*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights*(
_output_shapes
:??
?
Essd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights*
_output_shapes
: *
T0
?
Essd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/sub*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights*(
_output_shapes
:??
?
Assd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniformAddEssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights
?
&ssd_300_vgg/resblock2_1/conv_0/weights
VariableV2*
shape:??*
	container *(
_output_shapes
:??*
shared_name *
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights
?
-ssd_300_vgg/resblock2_1/conv_0/weights/AssignAssign&ssd_300_vgg/resblock2_1/conv_0/weightsAssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform*
validate_shape(*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights*
use_locking(*
T0
?
+ssd_300_vgg/resblock2_1/conv_0/weights/readIdentity&ssd_300_vgg/resblock2_1/conv_0/weights*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights*(
_output_shapes
:??*
T0
?
Fssd_300_vgg/resblock2_1/conv_0/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Gssd_300_vgg/resblock2_1/conv_0/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock2_1/conv_0/weights/read*
T0*
_output_shapes
: 
?
@ssd_300_vgg/resblock2_1/conv_0/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock2_1/conv_0/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock2_1/conv_0/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
7ssd_300_vgg/resblock2_1/conv_0/biases/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_0/biases
?
%ssd_300_vgg/resblock2_1/conv_0/biases
VariableV2*
shared_name *8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_0/biases*
shape:?*
_output_shapes	
:?*
	container *
dtype0
?
,ssd_300_vgg/resblock2_1/conv_0/biases/AssignAssign%ssd_300_vgg/resblock2_1/conv_0/biases7ssd_300_vgg/resblock2_1/conv_0/biases/Initializer/zeros*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_0/biases*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
*ssd_300_vgg/resblock2_1/conv_0/biases/readIdentity%ssd_300_vgg/resblock2_1/conv_0/biases*
_output_shapes	
:?*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_0/biases
}
,ssd_300_vgg/resblock2_1/conv_0/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
%ssd_300_vgg/resblock2_1/conv_0/Conv2DConv2Dssd_300_vgg/resblock2_1/Relu+ssd_300_vgg/resblock2_1/conv_0/weights/read*'
_output_shapes
:?*
	dilations
*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC
?
&ssd_300_vgg/resblock2_1/conv_0/BiasAddBiasAdd%ssd_300_vgg/resblock2_1/conv_0/Conv2D*ssd_300_vgg/resblock2_1/conv_0/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:?
?
;ssd_300_vgg/resblock2_1/batch_norm_1/beta/Initializer/zerosConst*
dtype0*
valueB?*    *
_output_shapes	
:?*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_1/beta
?
)ssd_300_vgg/resblock2_1/batch_norm_1/beta
VariableV2*
shape:?*
_output_shapes	
:?*
	container *<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_1/beta*
shared_name *
dtype0
?
0ssd_300_vgg/resblock2_1/batch_norm_1/beta/AssignAssign)ssd_300_vgg/resblock2_1/batch_norm_1/beta;ssd_300_vgg/resblock2_1/batch_norm_1/beta/Initializer/zeros*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_1/beta*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(
?
.ssd_300_vgg/resblock2_1/batch_norm_1/beta/readIdentity)ssd_300_vgg/resblock2_1/batch_norm_1/beta*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_1/beta*
T0*
_output_shapes	
:?
?
;ssd_300_vgg/resblock2_1/batch_norm_1/gamma/Initializer/onesConst*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_1/gamma*
valueB?*  ??*
dtype0
?
*ssd_300_vgg/resblock2_1/batch_norm_1/gamma
VariableV2*
shape:?*
	container *
shared_name *
dtype0*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_1/gamma*
_output_shapes	
:?
?
1ssd_300_vgg/resblock2_1/batch_norm_1/gamma/AssignAssign*ssd_300_vgg/resblock2_1/batch_norm_1/gamma;ssd_300_vgg/resblock2_1/batch_norm_1/gamma/Initializer/ones*
_output_shapes	
:?*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_1/gamma*
validate_shape(*
use_locking(
?
/ssd_300_vgg/resblock2_1/batch_norm_1/gamma/readIdentity*ssd_300_vgg/resblock2_1/batch_norm_1/gamma*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_1/gamma*
_output_shapes	
:?
?
Bssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean
?
0ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean
VariableV2*
	container *
shared_name *C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean*
dtype0*
_output_shapes	
:?*
shape:?
?
7ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/AssignAssign0ssd_300_vgg/resblock2_1/batch_norm_1/moving_meanBssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean
?
5ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/readIdentity0ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean*
T0*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean
?
Essd_300_vgg/resblock2_1/batch_norm_1/moving_variance/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance*
dtype0
?
4ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance
VariableV2*
dtype0*
	container *
shape:?*
shared_name *
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance
?
;ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/AssignAssign4ssd_300_vgg/resblock2_1/batch_norm_1/moving_varianceEssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/Initializer/ones*
_output_shapes	
:?*
T0*
validate_shape(*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance*
use_locking(
?
9ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/readIdentity4ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance*
T0
?
3ssd_300_vgg/resblock2_1/batch_norm_1/FusedBatchNormFusedBatchNorm&ssd_300_vgg/resblock2_1/conv_0/BiasAdd/ssd_300_vgg/resblock2_1/batch_norm_1/gamma/read.ssd_300_vgg/resblock2_1/batch_norm_1/beta/read5ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/read9ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/read*C
_output_shapes1
/:?:?:?:?:?*
epsilon%??'7*
data_formatNHWC*
is_training( *
T0
?
ssd_300_vgg/resblock2_1/Relu_1Relu3ssd_300_vgg/resblock2_1/batch_norm_1/FusedBatchNorm*'
_output_shapes
:?*
T0
?
Gssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights*%
valueB"            
?
Essd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *:??*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights
?
Essd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *:?=*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights
?
Ossd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformGssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/shape*
seed2 *(
_output_shapes
:??*
dtype0*

seed *9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights*
T0
?
Essd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/subSubEssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/maxEssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/min*
_output_shapes
: *9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights*
T0
?
Essd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/mulMulOssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/RandomUniformEssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/sub*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights*(
_output_shapes
:??
?
Assd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniformAddEssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/mulEssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform/min*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights*
T0*(
_output_shapes
:??
?
&ssd_300_vgg/resblock2_1/conv_1/weights
VariableV2*
	container *
shape:??*
dtype0*
shared_name *9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights*(
_output_shapes
:??
?
-ssd_300_vgg/resblock2_1/conv_1/weights/AssignAssign&ssd_300_vgg/resblock2_1/conv_1/weightsAssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform*(
_output_shapes
:??*
use_locking(*
T0*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights
?
+ssd_300_vgg/resblock2_1/conv_1/weights/readIdentity&ssd_300_vgg/resblock2_1/conv_1/weights*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights*(
_output_shapes
:??*
T0
?
Fssd_300_vgg/resblock2_1/conv_1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
_output_shapes
: *
dtype0
?
Gssd_300_vgg/resblock2_1/conv_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss+ssd_300_vgg/resblock2_1/conv_1/weights/read*
T0*
_output_shapes
: 
?
@ssd_300_vgg/resblock2_1/conv_1/kernel/Regularizer/l2_regularizerMulFssd_300_vgg/resblock2_1/conv_1/kernel/Regularizer/l2_regularizer/scaleGssd_300_vgg/resblock2_1/conv_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
7ssd_300_vgg/resblock2_1/conv_1/biases/Initializer/zerosConst*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_1/biases*
valueB?*    *
_output_shapes	
:?*
dtype0
?
%ssd_300_vgg/resblock2_1/conv_1/biases
VariableV2*
_output_shapes	
:?*
	container *
shape:?*
dtype0*
shared_name *8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_1/biases
?
,ssd_300_vgg/resblock2_1/conv_1/biases/AssignAssign%ssd_300_vgg/resblock2_1/conv_1/biases7ssd_300_vgg/resblock2_1/conv_1/biases/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_1/biases
?
*ssd_300_vgg/resblock2_1/conv_1/biases/readIdentity%ssd_300_vgg/resblock2_1/conv_1/biases*
T0*
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_1/biases
}
,ssd_300_vgg/resblock2_1/conv_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
%ssd_300_vgg/resblock2_1/conv_1/Conv2DConv2Dssd_300_vgg/resblock2_1/Relu_1+ssd_300_vgg/resblock2_1/conv_1/weights/read*
strides
*'
_output_shapes
:?*
paddingSAME*
data_formatNHWC*
	dilations
*
use_cudnn_on_gpu(*
T0
?
&ssd_300_vgg/resblock2_1/conv_1/BiasAddBiasAdd%ssd_300_vgg/resblock2_1/conv_1/Conv2D*ssd_300_vgg/resblock2_1/conv_1/biases/read*'
_output_shapes
:?*
data_formatNHWC*
T0
?
ssd_300_vgg/resblock2_1/addAdd&ssd_300_vgg/resblock2_1/conv_1/BiasAddssd_300_vgg/resblock2_0/add*'
_output_shapes
:?*
T0
?
<ssd_300_vgg/resblock_3_0/batch_norm_0/beta/Initializer/zerosConst*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/beta*
_output_shapes	
:?*
valueB?*    *
dtype0
?
*ssd_300_vgg/resblock_3_0/batch_norm_0/beta
VariableV2*
dtype0*
shape:?*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/beta*
shared_name *
	container *
_output_shapes	
:?
?
1ssd_300_vgg/resblock_3_0/batch_norm_0/beta/AssignAssign*ssd_300_vgg/resblock_3_0/batch_norm_0/beta<ssd_300_vgg/resblock_3_0/batch_norm_0/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/beta
?
/ssd_300_vgg/resblock_3_0/batch_norm_0/beta/readIdentity*ssd_300_vgg/resblock_3_0/batch_norm_0/beta*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/beta*
T0*
_output_shapes	
:?
?
<ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*
dtype0*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/gamma
?
+ssd_300_vgg/resblock_3_0/batch_norm_0/gamma
VariableV2*
dtype0*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/gamma*
shared_name *
shape:?*
	container *
_output_shapes	
:?
?
2ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/AssignAssign+ssd_300_vgg/resblock_3_0/batch_norm_0/gamma<ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/Initializer/ones*
_output_shapes	
:?*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/gamma*
use_locking(*
T0*
validate_shape(
?
0ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/readIdentity+ssd_300_vgg/resblock_3_0/batch_norm_0/gamma*
T0*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/gamma*
_output_shapes	
:?
?
Cssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean
?
1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean
VariableV2*
shared_name *
dtype0*
	container *
shape:?*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean*
_output_shapes	
:?
?
8ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/AssignAssign1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_meanCssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean
?
6ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/readIdentity1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean*
_output_shapes	
:?*
T0*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean
?
Fssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/Initializer/onesConst*
_output_shapes	
:?*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance*
dtype0*
valueB?*  ??
?
5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance
VariableV2*
shared_name *
_output_shapes	
:?*
shape:?*
	container *
dtype0*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance
?
<ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/AssignAssign5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_varianceFssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/Initializer/ones*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance
?
:ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/readIdentity5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance*
T0*
_output_shapes	
:?
?
4ssd_300_vgg/resblock_3_0/batch_norm_0/FusedBatchNormFusedBatchNormssd_300_vgg/resblock2_1/add0ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/read/ssd_300_vgg/resblock_3_0/batch_norm_0/beta/read6ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/read:ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/read*
is_training( *C
_output_shapes1
/:?:?:?:?:?*
data_formatNHWC*
epsilon%??'7*
T0
?
ssd_300_vgg/resblock_3_0/ReluRelu4ssd_300_vgg/resblock_3_0/batch_norm_0/FusedBatchNorm*'
_output_shapes
:?*
T0
?
%ssd_300_vgg/resblock_3_0/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
ssd_300_vgg/resblock_3_0/PadPadssd_300_vgg/resblock_3_0/Relu%ssd_300_vgg/resblock_3_0/Pad/paddings*'
_output_shapes
:?*
T0*
	Tpaddings0
?
Hssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/shapeConst*%
valueB"            *:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights*
dtype0*
_output_shapes
:
?
Fssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?[??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights
?
Fssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/maxConst*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights*
_output_shapes
: *
dtype0*
valueB
 *?[?<
?
Pssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/shape*
seed2 *
T0*(
_output_shapes
:??*

seed *:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights*
dtype0
?
Fssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/subSubFssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/maxFssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/min*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights*
_output_shapes
: *
T0
?
Fssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/mulMulPssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/sub*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights*(
_output_shapes
:??*
T0
?
Bssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniformAddFssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/mulFssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights*(
_output_shapes
:??
?
'ssd_300_vgg/resblock_3_0/conv_0/weights
VariableV2*
dtype0*
shared_name *
	container *:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights*(
_output_shapes
:??*
shape:??
?
.ssd_300_vgg/resblock_3_0/conv_0/weights/AssignAssign'ssd_300_vgg/resblock_3_0/conv_0/weightsBssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform*
use_locking(*
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights*(
_output_shapes
:??*
validate_shape(
?
,ssd_300_vgg/resblock_3_0/conv_0/weights/readIdentity'ssd_300_vgg/resblock_3_0/conv_0/weights*
T0*(
_output_shapes
:??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights
?
Gssd_300_vgg/resblock_3_0/conv_0/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Hssd_300_vgg/resblock_3_0/conv_0/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/resblock_3_0/conv_0/weights/read*
T0*
_output_shapes
: 
?
Assd_300_vgg/resblock_3_0/conv_0/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/resblock_3_0/conv_0/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/resblock_3_0/conv_0/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
8ssd_300_vgg/resblock_3_0/conv_0/biases/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_0/biases
?
&ssd_300_vgg/resblock_3_0/conv_0/biases
VariableV2*
	container *
dtype0*
shared_name *
_output_shapes	
:?*
shape:?*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_0/biases
?
-ssd_300_vgg/resblock_3_0/conv_0/biases/AssignAssign&ssd_300_vgg/resblock_3_0/conv_0/biases8ssd_300_vgg/resblock_3_0/conv_0/biases/Initializer/zeros*
_output_shapes	
:?*
validate_shape(*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_0/biases*
use_locking(
?
+ssd_300_vgg/resblock_3_0/conv_0/biases/readIdentity&ssd_300_vgg/resblock_3_0/conv_0/biases*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_0/biases*
_output_shapes	
:?
~
-ssd_300_vgg/resblock_3_0/conv_0/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
&ssd_300_vgg/resblock_3_0/conv_0/Conv2DConv2Dssd_300_vgg/resblock_3_0/Pad,ssd_300_vgg/resblock_3_0/conv_0/weights/read*
use_cudnn_on_gpu(*
strides
*'
_output_shapes
:?*
data_formatNHWC*
T0*
	dilations
*
paddingVALID
?
'ssd_300_vgg/resblock_3_0/conv_0/BiasAddBiasAdd&ssd_300_vgg/resblock_3_0/conv_0/Conv2D+ssd_300_vgg/resblock_3_0/conv_0/biases/read*'
_output_shapes
:?*
T0*
data_formatNHWC
?
'ssd_300_vgg/resblock_3_0/Pad_1/paddingsConst*9
value0B."                                 *
_output_shapes

:*
dtype0
?
ssd_300_vgg/resblock_3_0/Pad_1Padssd_300_vgg/resblock2_1/add'ssd_300_vgg/resblock_3_0/Pad_1/paddings*'
_output_shapes
:?*
T0*
	Tpaddings0
?
Kssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights*
dtype0
?
Issd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/minConst*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights*
_output_shapes
: *
dtype0*
valueB
 *???
?
Issd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *??=*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights
?
Sssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/RandomUniformRandomUniformKssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/shape*

seed *
dtype0*
T0*(
_output_shapes
:??*
seed2 *=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights
?
Issd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/subSubIssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/maxIssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/min*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights*
T0*
_output_shapes
: 
?
Issd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/mulMulSssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/RandomUniformIssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights*(
_output_shapes
:??
?
Essd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniformAddIssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/mulIssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform/min*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights*
T0*(
_output_shapes
:??
?
*ssd_300_vgg/resblock_3_0/conv_init/weights
VariableV2*
	container *
shared_name *
dtype0*
shape:??*(
_output_shapes
:??*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights
?
1ssd_300_vgg/resblock_3_0/conv_init/weights/AssignAssign*ssd_300_vgg/resblock_3_0/conv_init/weightsEssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform*
T0*(
_output_shapes
:??*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights*
validate_shape(*
use_locking(
?
/ssd_300_vgg/resblock_3_0/conv_init/weights/readIdentity*ssd_300_vgg/resblock_3_0/conv_init/weights*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights*(
_output_shapes
:??*
T0
?
Jssd_300_vgg/resblock_3_0/conv_init/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
_output_shapes
: *
dtype0
?
Kssd_300_vgg/resblock_3_0/conv_init/kernel/Regularizer/l2_regularizer/L2LossL2Loss/ssd_300_vgg/resblock_3_0/conv_init/weights/read*
T0*
_output_shapes
: 
?
Dssd_300_vgg/resblock_3_0/conv_init/kernel/Regularizer/l2_regularizerMulJssd_300_vgg/resblock_3_0/conv_init/kernel/Regularizer/l2_regularizer/scaleKssd_300_vgg/resblock_3_0/conv_init/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
;ssd_300_vgg/resblock_3_0/conv_init/biases/Initializer/zerosConst*<
_class2
0.loc:@ssd_300_vgg/resblock_3_0/conv_init/biases*
dtype0*
_output_shapes	
:?*
valueB?*    
?
)ssd_300_vgg/resblock_3_0/conv_init/biases
VariableV2*
	container *
dtype0*<
_class2
0.loc:@ssd_300_vgg/resblock_3_0/conv_init/biases*
_output_shapes	
:?*
shared_name *
shape:?
?
0ssd_300_vgg/resblock_3_0/conv_init/biases/AssignAssign)ssd_300_vgg/resblock_3_0/conv_init/biases;ssd_300_vgg/resblock_3_0/conv_init/biases/Initializer/zeros*
validate_shape(*<
_class2
0.loc:@ssd_300_vgg/resblock_3_0/conv_init/biases*
_output_shapes	
:?*
use_locking(*
T0
?
.ssd_300_vgg/resblock_3_0/conv_init/biases/readIdentity)ssd_300_vgg/resblock_3_0/conv_init/biases*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock_3_0/conv_init/biases*
_output_shapes	
:?
?
0ssd_300_vgg/resblock_3_0/conv_init/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
)ssd_300_vgg/resblock_3_0/conv_init/Conv2DConv2Dssd_300_vgg/resblock_3_0/Pad_1/ssd_300_vgg/resblock_3_0/conv_init/weights/read*
data_formatNHWC*
T0*
	dilations
*'
_output_shapes
:?*
paddingVALID*
strides
*
use_cudnn_on_gpu(
?
*ssd_300_vgg/resblock_3_0/conv_init/BiasAddBiasAdd)ssd_300_vgg/resblock_3_0/conv_init/Conv2D.ssd_300_vgg/resblock_3_0/conv_init/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:?
?
<ssd_300_vgg/resblock_3_0/batch_norm_1/beta/Initializer/zerosConst*
valueB?*    *
dtype0*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/beta*
_output_shapes	
:?
?
*ssd_300_vgg/resblock_3_0/batch_norm_1/beta
VariableV2*
	container *
shape:?*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/beta*
shared_name *
dtype0*
_output_shapes	
:?
?
1ssd_300_vgg/resblock_3_0/batch_norm_1/beta/AssignAssign*ssd_300_vgg/resblock_3_0/batch_norm_1/beta<ssd_300_vgg/resblock_3_0/batch_norm_1/beta/Initializer/zeros*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/beta*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
/ssd_300_vgg/resblock_3_0/batch_norm_1/beta/readIdentity*ssd_300_vgg/resblock_3_0/batch_norm_1/beta*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/beta*
_output_shapes	
:?*
T0
?
<ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/Initializer/onesConst*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/gamma*
valueB?*  ??*
_output_shapes	
:?*
dtype0
?
+ssd_300_vgg/resblock_3_0/batch_norm_1/gamma
VariableV2*
dtype0*
shared_name *
	container *
_output_shapes	
:?*
shape:?*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/gamma
?
2ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/AssignAssign+ssd_300_vgg/resblock_3_0/batch_norm_1/gamma<ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/Initializer/ones*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/gamma
?
0ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/readIdentity+ssd_300_vgg/resblock_3_0/batch_norm_1/gamma*
_output_shapes	
:?*
T0*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/gamma
?
Cssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *
_output_shapes	
:?*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean
?
1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean
VariableV2*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean*
shape:?*
	container *
dtype0*
shared_name *
_output_shapes	
:?
?
8ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/AssignAssign1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_meanCssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/Initializer/zeros*
use_locking(*
_output_shapes	
:?*
T0*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean*
validate_shape(
?
6ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/readIdentity1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean*
_output_shapes	
:?*
T0*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean
?
Fssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/Initializer/onesConst*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance*
_output_shapes	
:?*
valueB?*  ??*
dtype0
?
5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance
VariableV2*
_output_shapes	
:?*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance*
shape:?*
dtype0*
shared_name *
	container 
?
<ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/AssignAssign5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_varianceFssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/Initializer/ones*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(
?
:ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/readIdentity5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance*
T0*
_output_shapes	
:?*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance
?
4ssd_300_vgg/resblock_3_0/batch_norm_1/FusedBatchNormFusedBatchNorm'ssd_300_vgg/resblock_3_0/conv_0/BiasAdd0ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/read/ssd_300_vgg/resblock_3_0/batch_norm_1/beta/read6ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/read:ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/read*
epsilon%??'7*
data_formatNHWC*C
_output_shapes1
/:?:?:?:?:?*
is_training( *
T0
?
ssd_300_vgg/resblock_3_0/Relu_1Relu4ssd_300_vgg/resblock_3_0/batch_norm_1/FusedBatchNorm*'
_output_shapes
:?*
T0
?
Hssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights*
dtype0*%
valueB"            *
_output_shapes
:
?
Fssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/minConst*
dtype0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights*
valueB
 *?Ѽ*
_output_shapes
: 
?
Fssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/maxConst*
valueB
 *??<*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights*
_output_shapes
: *
dtype0
?
Pssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/shape*

seed *
seed2 *(
_output_shapes
:??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights*
T0*
dtype0
?
Fssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/subSubFssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/maxFssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/min*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights*
T0*
_output_shapes
: 
?
Fssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/mulMulPssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights
?
Bssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniformAddFssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/mulFssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform/min*(
_output_shapes
:??*
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights
?
'ssd_300_vgg/resblock_3_0/conv_1/weights
VariableV2*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights*(
_output_shapes
:??*
shared_name *
	container *
dtype0*
shape:??
?
.ssd_300_vgg/resblock_3_0/conv_1/weights/AssignAssign'ssd_300_vgg/resblock_3_0/conv_1/weightsBssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform*(
_output_shapes
:??*
use_locking(*
validate_shape(*
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights
?
,ssd_300_vgg/resblock_3_0/conv_1/weights/readIdentity'ssd_300_vgg/resblock_3_0/conv_1/weights*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights*(
_output_shapes
:??*
T0
?
Gssd_300_vgg/resblock_3_0/conv_1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:
?
Hssd_300_vgg/resblock_3_0/conv_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/resblock_3_0/conv_1/weights/read*
_output_shapes
: *
T0
?
Assd_300_vgg/resblock_3_0/conv_1/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/resblock_3_0/conv_1/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/resblock_3_0/conv_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
8ssd_300_vgg/resblock_3_0/conv_1/biases/Initializer/zerosConst*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_1/biases*
valueB?*    *
_output_shapes	
:?*
dtype0
?
&ssd_300_vgg/resblock_3_0/conv_1/biases
VariableV2*
_output_shapes	
:?*
dtype0*
shape:?*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_1/biases*
	container *
shared_name 
?
-ssd_300_vgg/resblock_3_0/conv_1/biases/AssignAssign&ssd_300_vgg/resblock_3_0/conv_1/biases8ssd_300_vgg/resblock_3_0/conv_1/biases/Initializer/zeros*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_1/biases*
_output_shapes	
:?*
T0*
use_locking(*
validate_shape(
?
+ssd_300_vgg/resblock_3_0/conv_1/biases/readIdentity&ssd_300_vgg/resblock_3_0/conv_1/biases*
T0*
_output_shapes	
:?*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_1/biases
~
-ssd_300_vgg/resblock_3_0/conv_1/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
&ssd_300_vgg/resblock_3_0/conv_1/Conv2DConv2Dssd_300_vgg/resblock_3_0/Relu_1,ssd_300_vgg/resblock_3_0/conv_1/weights/read*
data_formatNHWC*
strides
*
	dilations
*
use_cudnn_on_gpu(*
T0*
paddingSAME*'
_output_shapes
:?
?
'ssd_300_vgg/resblock_3_0/conv_1/BiasAddBiasAdd&ssd_300_vgg/resblock_3_0/conv_1/Conv2D+ssd_300_vgg/resblock_3_0/conv_1/biases/read*
T0*'
_output_shapes
:?*
data_formatNHWC
?
ssd_300_vgg/resblock_3_0/addAdd'ssd_300_vgg/resblock_3_0/conv_1/BiasAdd*ssd_300_vgg/resblock_3_0/conv_init/BiasAdd*
T0*'
_output_shapes
:?
?
<ssd_300_vgg/resblock_3_1/batch_norm_0/beta/Initializer/zerosConst*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/beta*
valueB?*    *
_output_shapes	
:?*
dtype0
?
*ssd_300_vgg/resblock_3_1/batch_norm_0/beta
VariableV2*
	container *
_output_shapes	
:?*
shape:?*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/beta*
dtype0*
shared_name 
?
1ssd_300_vgg/resblock_3_1/batch_norm_0/beta/AssignAssign*ssd_300_vgg/resblock_3_1/batch_norm_0/beta<ssd_300_vgg/resblock_3_1/batch_norm_0/beta/Initializer/zeros*
use_locking(*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/beta*
T0*
validate_shape(
?
/ssd_300_vgg/resblock_3_1/batch_norm_0/beta/readIdentity*ssd_300_vgg/resblock_3_1/batch_norm_0/beta*
_output_shapes	
:?*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/beta
?
<ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/Initializer/onesConst*
_output_shapes	
:?*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/gamma*
valueB?*  ??*
dtype0
?
+ssd_300_vgg/resblock_3_1/batch_norm_0/gamma
VariableV2*
shared_name *>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/gamma*
shape:?*
dtype0*
	container *
_output_shapes	
:?
?
2ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/AssignAssign+ssd_300_vgg/resblock_3_1/batch_norm_0/gamma<ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/Initializer/ones*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/gamma
?
0ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/readIdentity+ssd_300_vgg/resblock_3_1/batch_norm_0/gamma*
_output_shapes	
:?*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/gamma*
T0
?
Cssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*
dtype0*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean
?
1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean
VariableV2*
shape:?*
dtype0*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean*
_output_shapes	
:?*
	container *
shared_name 
?
8ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/AssignAssign1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_meanCssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/Initializer/zeros*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0
?
6ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/readIdentity1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean*
_output_shapes	
:?*
T0
?
Fssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/Initializer/onesConst*
_output_shapes	
:?*
dtype0*
valueB?*  ??*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance
?
5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance
VariableV2*
shared_name *
	container *
_output_shapes	
:?*
shape:?*
dtype0*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance
?
<ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/AssignAssign5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_varianceFssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/Initializer/ones*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
:ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/readIdentity5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance*
T0*
_output_shapes	
:?
?
4ssd_300_vgg/resblock_3_1/batch_norm_0/FusedBatchNormFusedBatchNormssd_300_vgg/resblock_3_0/add0ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/read/ssd_300_vgg/resblock_3_1/batch_norm_0/beta/read6ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/read:ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/read*
is_training( *
T0*
epsilon%??'7*C
_output_shapes1
/:?:?:?:?:?*
data_formatNHWC
?
ssd_300_vgg/resblock_3_1/ReluRelu4ssd_300_vgg/resblock_3_1/batch_norm_0/FusedBatchNorm*'
_output_shapes
:?*
T0
?
Hssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/shapeConst*
dtype0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights*
_output_shapes
:*%
valueB"            
?
Fssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/minConst*
dtype0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights*
_output_shapes
: *
valueB
 *?Ѽ
?
Fssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/maxConst*
valueB
 *??<*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights*
dtype0
?
Pssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/shape*

seed *
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights*
dtype0*(
_output_shapes
:??*
seed2 
?
Fssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/subSubFssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/maxFssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights
?
Fssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/mulMulPssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights
?
Bssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniformAddFssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/mulFssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights
?
'ssd_300_vgg/resblock_3_1/conv_0/weights
VariableV2*
	container *
dtype0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights*
shape:??*(
_output_shapes
:??*
shared_name 
?
.ssd_300_vgg/resblock_3_1/conv_0/weights/AssignAssign'ssd_300_vgg/resblock_3_1/conv_0/weightsBssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform*
validate_shape(*
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights*(
_output_shapes
:??*
use_locking(
?
,ssd_300_vgg/resblock_3_1/conv_0/weights/readIdentity'ssd_300_vgg/resblock_3_1/conv_0/weights*(
_output_shapes
:??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights*
T0
?
Gssd_300_vgg/resblock_3_1/conv_0/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *o:*
dtype0
?
Hssd_300_vgg/resblock_3_1/conv_0/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/resblock_3_1/conv_0/weights/read*
T0*
_output_shapes
: 
?
Assd_300_vgg/resblock_3_1/conv_0/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/resblock_3_1/conv_0/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/resblock_3_1/conv_0/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
8ssd_300_vgg/resblock_3_1/conv_0/biases/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_0/biases*
valueB?*    
?
&ssd_300_vgg/resblock_3_1/conv_0/biases
VariableV2*
_output_shapes	
:?*
dtype0*
	container *
shape:?*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_0/biases*
shared_name 
?
-ssd_300_vgg/resblock_3_1/conv_0/biases/AssignAssign&ssd_300_vgg/resblock_3_1/conv_0/biases8ssd_300_vgg/resblock_3_1/conv_0/biases/Initializer/zeros*
T0*
use_locking(*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_0/biases*
_output_shapes	
:?
?
+ssd_300_vgg/resblock_3_1/conv_0/biases/readIdentity&ssd_300_vgg/resblock_3_1/conv_0/biases*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_0/biases*
_output_shapes	
:?
~
-ssd_300_vgg/resblock_3_1/conv_0/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
&ssd_300_vgg/resblock_3_1/conv_0/Conv2DConv2Dssd_300_vgg/resblock_3_1/Relu,ssd_300_vgg/resblock_3_1/conv_0/weights/read*
	dilations
*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*'
_output_shapes
:?*
paddingSAME*
T0
?
'ssd_300_vgg/resblock_3_1/conv_0/BiasAddBiasAdd&ssd_300_vgg/resblock_3_1/conv_0/Conv2D+ssd_300_vgg/resblock_3_1/conv_0/biases/read*'
_output_shapes
:?*
T0*
data_formatNHWC
?
<ssd_300_vgg/resblock_3_1/batch_norm_1/beta/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/beta
?
*ssd_300_vgg/resblock_3_1/batch_norm_1/beta
VariableV2*
	container *
_output_shapes	
:?*
shared_name *
shape:?*
dtype0*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/beta
?
1ssd_300_vgg/resblock_3_1/batch_norm_1/beta/AssignAssign*ssd_300_vgg/resblock_3_1/batch_norm_1/beta<ssd_300_vgg/resblock_3_1/batch_norm_1/beta/Initializer/zeros*
T0*
use_locking(*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/beta*
_output_shapes	
:?
?
/ssd_300_vgg/resblock_3_1/batch_norm_1/beta/readIdentity*ssd_300_vgg/resblock_3_1/batch_norm_1/beta*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/beta*
_output_shapes	
:?
?
<ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/gamma
?
+ssd_300_vgg/resblock_3_1/batch_norm_1/gamma
VariableV2*
_output_shapes	
:?*
shape:?*
shared_name *
dtype0*
	container *>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/gamma
?
2ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/AssignAssign+ssd_300_vgg/resblock_3_1/batch_norm_1/gamma<ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/Initializer/ones*
validate_shape(*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/gamma*
_output_shapes	
:?*
T0*
use_locking(
?
0ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/readIdentity+ssd_300_vgg/resblock_3_1/batch_norm_1/gamma*
_output_shapes	
:?*
T0*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/gamma
?
Cssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/Initializer/zerosConst*
valueB?*    *D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean*
dtype0*
_output_shapes	
:?
?
1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean
VariableV2*
shared_name *
	container *
dtype0*
shape:?*
_output_shapes	
:?*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean
?
8ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/AssignAssign1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_meanCssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/Initializer/zeros*
validate_shape(*
use_locking(*
T0*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean*
_output_shapes	
:?
?
6ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/readIdentity1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean*
T0*
_output_shapes	
:?*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean
?
Fssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/Initializer/onesConst*
_output_shapes	
:?*
dtype0*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance*
valueB?*  ??
?
5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance
VariableV2*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance*
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *
	container 
?
<ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/AssignAssign5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_varianceFssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/Initializer/ones*
validate_shape(*
use_locking(*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance*
T0*
_output_shapes	
:?
?
:ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/readIdentity5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance*
T0*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance*
_output_shapes	
:?
?
4ssd_300_vgg/resblock_3_1/batch_norm_1/FusedBatchNormFusedBatchNorm'ssd_300_vgg/resblock_3_1/conv_0/BiasAdd0ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/read/ssd_300_vgg/resblock_3_1/batch_norm_1/beta/read6ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/read:ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/read*
is_training( *C
_output_shapes1
/:?:?:?:?:?*
epsilon%??'7*
T0*
data_formatNHWC
?
ssd_300_vgg/resblock_3_1/Relu_1Relu4ssd_300_vgg/resblock_3_1/batch_norm_1/FusedBatchNorm*
T0*'
_output_shapes
:?
?
Hssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights*
dtype0
?
Fssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/minConst*
valueB
 *?Ѽ*
dtype0*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights
?
Fssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/maxConst*
dtype0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights*
valueB
 *??<*
_output_shapes
: 
?
Pssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/shape*(
_output_shapes
:??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights*
dtype0*
seed2 *

seed *
T0
?
Fssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/subSubFssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/maxFssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights
?
Fssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/mulMulPssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights*(
_output_shapes
:??
?
Bssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniformAddFssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/mulFssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform/min*(
_output_shapes
:??*
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights
?
'ssd_300_vgg/resblock_3_1/conv_1/weights
VariableV2*(
_output_shapes
:??*
dtype0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights*
shape:??*
	container *
shared_name 
?
.ssd_300_vgg/resblock_3_1/conv_1/weights/AssignAssign'ssd_300_vgg/resblock_3_1/conv_1/weightsBssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform*
use_locking(*(
_output_shapes
:??*
T0*
validate_shape(*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights
?
,ssd_300_vgg/resblock_3_1/conv_1/weights/readIdentity'ssd_300_vgg/resblock_3_1/conv_1/weights*
T0*(
_output_shapes
:??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights
?
Gssd_300_vgg/resblock_3_1/conv_1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:
?
Hssd_300_vgg/resblock_3_1/conv_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/resblock_3_1/conv_1/weights/read*
_output_shapes
: *
T0
?
Assd_300_vgg/resblock_3_1/conv_1/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/resblock_3_1/conv_1/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/resblock_3_1/conv_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
8ssd_300_vgg/resblock_3_1/conv_1/biases/Initializer/zerosConst*
valueB?*    *9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_1/biases*
dtype0*
_output_shapes	
:?
?
&ssd_300_vgg/resblock_3_1/conv_1/biases
VariableV2*
shape:?*
	container *9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_1/biases*
_output_shapes	
:?*
shared_name *
dtype0
?
-ssd_300_vgg/resblock_3_1/conv_1/biases/AssignAssign&ssd_300_vgg/resblock_3_1/conv_1/biases8ssd_300_vgg/resblock_3_1/conv_1/biases/Initializer/zeros*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_1/biases
?
+ssd_300_vgg/resblock_3_1/conv_1/biases/readIdentity&ssd_300_vgg/resblock_3_1/conv_1/biases*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_1/biases*
_output_shapes	
:?
~
-ssd_300_vgg/resblock_3_1/conv_1/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
&ssd_300_vgg/resblock_3_1/conv_1/Conv2DConv2Dssd_300_vgg/resblock_3_1/Relu_1,ssd_300_vgg/resblock_3_1/conv_1/weights/read*
strides
*
data_formatNHWC*'
_output_shapes
:?*
	dilations
*
T0*
use_cudnn_on_gpu(*
paddingSAME
?
'ssd_300_vgg/resblock_3_1/conv_1/BiasAddBiasAdd&ssd_300_vgg/resblock_3_1/conv_1/Conv2D+ssd_300_vgg/resblock_3_1/conv_1/biases/read*'
_output_shapes
:?*
data_formatNHWC*
T0
?
ssd_300_vgg/resblock_3_1/addAdd'ssd_300_vgg/resblock_3_1/conv_1/BiasAddssd_300_vgg/resblock_3_0/add*
T0*'
_output_shapes
:?
?
<ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/shapeConst*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*%
valueB"         ?   *
_output_shapes
:*
dtype0
?
:ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *?Kƽ*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights
?
:ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/maxConst*
dtype0*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*
valueB
 *?K?=*
_output_shapes
: 
?
Dssd_300_vgg/conv8_1/weights/Initializer/random_uniform/RandomUniformRandomUniform<ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/shape*
dtype0*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*
T0*
seed2 *(
_output_shapes
:??*

seed 
?
:ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/subSub:ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/max:ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/min*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*
_output_shapes
: *
T0
?
:ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/mulMulDssd_300_vgg/conv8_1/weights/Initializer/random_uniform/RandomUniform:ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*
T0
?
6ssd_300_vgg/conv8_1/weights/Initializer/random_uniformAdd:ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/mul:ssd_300_vgg/conv8_1/weights/Initializer/random_uniform/min*
T0*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*(
_output_shapes
:??
?
ssd_300_vgg/conv8_1/weights
VariableV2*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*
shared_name *
	container *(
_output_shapes
:??*
shape:??*
dtype0
?
"ssd_300_vgg/conv8_1/weights/AssignAssignssd_300_vgg/conv8_1/weights6ssd_300_vgg/conv8_1/weights/Initializer/random_uniform*
validate_shape(*(
_output_shapes
:??*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*
use_locking(*
T0
?
 ssd_300_vgg/conv8_1/weights/readIdentityssd_300_vgg/conv8_1/weights*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*
T0*(
_output_shapes
:??
?
;ssd_300_vgg/conv8_1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
_output_shapes
: *
dtype0
?
<ssd_300_vgg/conv8_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss ssd_300_vgg/conv8_1/weights/read*
_output_shapes
: *
T0
?
5ssd_300_vgg/conv8_1/kernel/Regularizer/l2_regularizerMul;ssd_300_vgg/conv8_1/kernel/Regularizer/l2_regularizer/scale<ssd_300_vgg/conv8_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
,ssd_300_vgg/conv8_1/biases/Initializer/zerosConst*-
_class#
!loc:@ssd_300_vgg/conv8_1/biases*
_output_shapes	
:?*
dtype0*
valueB?*    
?
ssd_300_vgg/conv8_1/biases
VariableV2*
_output_shapes	
:?*-
_class#
!loc:@ssd_300_vgg/conv8_1/biases*
shared_name *
shape:?*
	container *
dtype0
?
!ssd_300_vgg/conv8_1/biases/AssignAssignssd_300_vgg/conv8_1/biases,ssd_300_vgg/conv8_1/biases/Initializer/zeros*
_output_shapes	
:?*
validate_shape(*-
_class#
!loc:@ssd_300_vgg/conv8_1/biases*
T0*
use_locking(
?
ssd_300_vgg/conv8_1/biases/readIdentityssd_300_vgg/conv8_1/biases*
_output_shapes	
:?*-
_class#
!loc:@ssd_300_vgg/conv8_1/biases*
T0
r
!ssd_300_vgg/conv8_1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
ssd_300_vgg/conv8_1/Conv2DConv2Dssd_300_vgg/resblock_3_1/add ssd_300_vgg/conv8_1/weights/read*
strides
*
data_formatNHWC*
	dilations
*
T0*'
_output_shapes
:?*
use_cudnn_on_gpu(*
paddingSAME
?
ssd_300_vgg/conv8_1/BiasAddBiasAddssd_300_vgg/conv8_1/Conv2Dssd_300_vgg/conv8_1/biases/read*
T0*'
_output_shapes
:?*
data_formatNHWC
i
ssd_300_vgg/Relu_1Relussd_300_vgg/conv8_1/BiasAdd*'
_output_shapes
:?*
T0
?
<ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/shapeConst*%
valueB"      ?   ?   *
dtype0*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights*
_output_shapes
:
?
:ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/minConst*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights*
valueB
 *?Q?*
dtype0*
_output_shapes
: 
?
:ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/maxConst*
dtype0*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights*
valueB
 *?Q=*
_output_shapes
: 
?
Dssd_300_vgg/conv8_2/weights/Initializer/random_uniform/RandomUniformRandomUniform<ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/shape*(
_output_shapes
:??*
T0*
seed2 *
dtype0*

seed *.
_class$
" loc:@ssd_300_vgg/conv8_2/weights
?
:ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/subSub:ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/max:ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/min*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights*
_output_shapes
: *
T0
?
:ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/mulMulDssd_300_vgg/conv8_2/weights/Initializer/random_uniform/RandomUniform:ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights
?
6ssd_300_vgg/conv8_2/weights/Initializer/random_uniformAdd:ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/mul:ssd_300_vgg/conv8_2/weights/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights
?
ssd_300_vgg/conv8_2/weights
VariableV2*(
_output_shapes
:??*
shape:??*
	container *
dtype0*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights*
shared_name 
?
"ssd_300_vgg/conv8_2/weights/AssignAssignssd_300_vgg/conv8_2/weights6ssd_300_vgg/conv8_2/weights/Initializer/random_uniform*
T0*(
_output_shapes
:??*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights*
validate_shape(*
use_locking(
?
 ssd_300_vgg/conv8_2/weights/readIdentityssd_300_vgg/conv8_2/weights*
T0*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights*(
_output_shapes
:??
?
;ssd_300_vgg/conv8_2/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:
?
<ssd_300_vgg/conv8_2/kernel/Regularizer/l2_regularizer/L2LossL2Loss ssd_300_vgg/conv8_2/weights/read*
_output_shapes
: *
T0
?
5ssd_300_vgg/conv8_2/kernel/Regularizer/l2_regularizerMul;ssd_300_vgg/conv8_2/kernel/Regularizer/l2_regularizer/scale<ssd_300_vgg/conv8_2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
,ssd_300_vgg/conv8_2/biases/Initializer/zerosConst*
_output_shapes	
:?*-
_class#
!loc:@ssd_300_vgg/conv8_2/biases*
dtype0*
valueB?*    
?
ssd_300_vgg/conv8_2/biases
VariableV2*
	container *-
_class#
!loc:@ssd_300_vgg/conv8_2/biases*
dtype0*
_output_shapes	
:?*
shape:?*
shared_name 
?
!ssd_300_vgg/conv8_2/biases/AssignAssignssd_300_vgg/conv8_2/biases,ssd_300_vgg/conv8_2/biases/Initializer/zeros*
validate_shape(*
T0*
_output_shapes	
:?*-
_class#
!loc:@ssd_300_vgg/conv8_2/biases*
use_locking(
?
ssd_300_vgg/conv8_2/biases/readIdentityssd_300_vgg/conv8_2/biases*
_output_shapes	
:?*-
_class#
!loc:@ssd_300_vgg/conv8_2/biases*
T0
r
!ssd_300_vgg/conv8_2/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
ssd_300_vgg/conv8_2/Conv2DConv2Dssd_300_vgg/Relu_1 ssd_300_vgg/conv8_2/weights/read*
data_formatNHWC*
paddingSAME*'
_output_shapes
:?*
use_cudnn_on_gpu(*
	dilations
*
strides
*
T0
?
ssd_300_vgg/conv8_2/BiasAddBiasAddssd_300_vgg/conv8_2/Conv2Dssd_300_vgg/conv8_2/biases/read*'
_output_shapes
:?*
T0*
data_formatNHWC
i
ssd_300_vgg/Relu_2Relussd_300_vgg/conv8_2/BiasAdd*'
_output_shapes
:?*
T0
?
<ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*
dtype0*%
valueB"      ?   ?   
?
:ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/minConst*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*
valueB
 *q??*
dtype0*
_output_shapes
: 
?
:ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *q?>*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights
?
Dssd_300_vgg/conv9_1/weights/Initializer/random_uniform/RandomUniformRandomUniform<ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/shape*
dtype0*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*

seed *
seed2 *(
_output_shapes
:??*
T0
?
:ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/subSub:ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/max:ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/min*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*
T0*
_output_shapes
: 
?
:ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/mulMulDssd_300_vgg/conv9_1/weights/Initializer/random_uniform/RandomUniform:ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*(
_output_shapes
:??
?
6ssd_300_vgg/conv9_1/weights/Initializer/random_uniformAdd:ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/mul:ssd_300_vgg/conv9_1/weights/Initializer/random_uniform/min*
T0*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*(
_output_shapes
:??
?
ssd_300_vgg/conv9_1/weights
VariableV2*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*
shape:??*
	container *
dtype0*(
_output_shapes
:??*
shared_name 
?
"ssd_300_vgg/conv9_1/weights/AssignAssignssd_300_vgg/conv9_1/weights6ssd_300_vgg/conv9_1/weights/Initializer/random_uniform*(
_output_shapes
:??*
validate_shape(*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*
use_locking(*
T0
?
 ssd_300_vgg/conv9_1/weights/readIdentityssd_300_vgg/conv9_1/weights*(
_output_shapes
:??*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*
T0
?
;ssd_300_vgg/conv9_1/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:
?
<ssd_300_vgg/conv9_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss ssd_300_vgg/conv9_1/weights/read*
T0*
_output_shapes
: 
?
5ssd_300_vgg/conv9_1/kernel/Regularizer/l2_regularizerMul;ssd_300_vgg/conv9_1/kernel/Regularizer/l2_regularizer/scale<ssd_300_vgg/conv9_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
,ssd_300_vgg/conv9_1/biases/Initializer/zerosConst*-
_class#
!loc:@ssd_300_vgg/conv9_1/biases*
_output_shapes	
:?*
valueB?*    *
dtype0
?
ssd_300_vgg/conv9_1/biases
VariableV2*
dtype0*
shared_name *
shape:?*
	container *
_output_shapes	
:?*-
_class#
!loc:@ssd_300_vgg/conv9_1/biases
?
!ssd_300_vgg/conv9_1/biases/AssignAssignssd_300_vgg/conv9_1/biases,ssd_300_vgg/conv9_1/biases/Initializer/zeros*
T0*
use_locking(*
validate_shape(*-
_class#
!loc:@ssd_300_vgg/conv9_1/biases*
_output_shapes	
:?
?
ssd_300_vgg/conv9_1/biases/readIdentityssd_300_vgg/conv9_1/biases*-
_class#
!loc:@ssd_300_vgg/conv9_1/biases*
T0*
_output_shapes	
:?
r
!ssd_300_vgg/conv9_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
ssd_300_vgg/conv9_1/Conv2DConv2Dssd_300_vgg/Relu_2 ssd_300_vgg/conv9_1/weights/read*
strides
*
	dilations
*
use_cudnn_on_gpu(*
paddingSAME*
T0*'
_output_shapes
:?*
data_formatNHWC
?
ssd_300_vgg/conv9_1/BiasAddBiasAddssd_300_vgg/conv9_1/Conv2Dssd_300_vgg/conv9_1/biases/read*'
_output_shapes
:?*
T0*
data_formatNHWC
i
ssd_300_vgg/Relu_3Relussd_300_vgg/conv9_1/BiasAdd*
T0*'
_output_shapes
:?
?
ssd_300_vgg/Pad_1/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0
?
ssd_300_vgg/Pad_1Padssd_300_vgg/Relu_3ssd_300_vgg/Pad_1/paddings*
T0*'
_output_shapes
:		?*
	Tpaddings0
?
<ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/shapeConst*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights*
dtype0*%
valueB"      ?   ?   *
_output_shapes
:
?
:ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/minConst*
valueB
 *?Q?*
_output_shapes
: *.
_class$
" loc:@ssd_300_vgg/conv9_2/weights*
dtype0
?
:ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/maxConst*
valueB
 *?Q=*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights*
dtype0*
_output_shapes
: 
?
Dssd_300_vgg/conv9_2/weights/Initializer/random_uniform/RandomUniformRandomUniform<ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/shape*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights*
seed2 *(
_output_shapes
:??*

seed *
dtype0*
T0
?
:ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/subSub:ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/max:ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/min*
_output_shapes
: *.
_class$
" loc:@ssd_300_vgg/conv9_2/weights*
T0
?
:ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/mulMulDssd_300_vgg/conv9_2/weights/Initializer/random_uniform/RandomUniform:ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/sub*
T0*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights*(
_output_shapes
:??
?
6ssd_300_vgg/conv9_2/weights/Initializer/random_uniformAdd:ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/mul:ssd_300_vgg/conv9_2/weights/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights
?
ssd_300_vgg/conv9_2/weights
VariableV2*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights*
	container *
dtype0*
shared_name *(
_output_shapes
:??*
shape:??
?
"ssd_300_vgg/conv9_2/weights/AssignAssignssd_300_vgg/conv9_2/weights6ssd_300_vgg/conv9_2/weights/Initializer/random_uniform*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights*(
_output_shapes
:??*
use_locking(*
validate_shape(*
T0
?
 ssd_300_vgg/conv9_2/weights/readIdentityssd_300_vgg/conv9_2/weights*(
_output_shapes
:??*
T0*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights
?
;ssd_300_vgg/conv9_2/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
?
<ssd_300_vgg/conv9_2/kernel/Regularizer/l2_regularizer/L2LossL2Loss ssd_300_vgg/conv9_2/weights/read*
_output_shapes
: *
T0
?
5ssd_300_vgg/conv9_2/kernel/Regularizer/l2_regularizerMul;ssd_300_vgg/conv9_2/kernel/Regularizer/l2_regularizer/scale<ssd_300_vgg/conv9_2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
,ssd_300_vgg/conv9_2/biases/Initializer/zerosConst*
valueB?*    *-
_class#
!loc:@ssd_300_vgg/conv9_2/biases*
dtype0*
_output_shapes	
:?
?
ssd_300_vgg/conv9_2/biases
VariableV2*
	container *
dtype0*
shared_name *-
_class#
!loc:@ssd_300_vgg/conv9_2/biases*
_output_shapes	
:?*
shape:?
?
!ssd_300_vgg/conv9_2/biases/AssignAssignssd_300_vgg/conv9_2/biases,ssd_300_vgg/conv9_2/biases/Initializer/zeros*-
_class#
!loc:@ssd_300_vgg/conv9_2/biases*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(
?
ssd_300_vgg/conv9_2/biases/readIdentityssd_300_vgg/conv9_2/biases*-
_class#
!loc:@ssd_300_vgg/conv9_2/biases*
_output_shapes	
:?*
T0
r
!ssd_300_vgg/conv9_2/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
ssd_300_vgg/conv9_2/Conv2DConv2Dssd_300_vgg/Pad_1 ssd_300_vgg/conv9_2/weights/read*'
_output_shapes
:?*
data_formatNHWC*
T0*
strides
*
use_cudnn_on_gpu(*
paddingVALID*
	dilations

?
ssd_300_vgg/conv9_2/BiasAddBiasAddssd_300_vgg/conv9_2/Conv2Dssd_300_vgg/conv9_2/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:?
i
ssd_300_vgg/Relu_4Relussd_300_vgg/conv9_2/BiasAdd*'
_output_shapes
:?*
T0
?
=ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*%
valueB"      ?   ?   
?
;ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/minConst*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*
valueB
 *q??*
dtype0*
_output_shapes
: 
?
;ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/maxConst*
valueB
 *q?>*
dtype0*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*
_output_shapes
: 
?
Essd_300_vgg/conv10_1/weights/Initializer/random_uniform/RandomUniformRandomUniform=ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/shape*
dtype0*

seed *
T0*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*
seed2 *(
_output_shapes
:??
?
;ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/subSub;ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/max;ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/min*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*
_output_shapes
: *
T0
?
;ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/mulMulEssd_300_vgg/conv10_1/weights/Initializer/random_uniform/RandomUniform;ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*
T0
?
7ssd_300_vgg/conv10_1/weights/Initializer/random_uniformAdd;ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/mul;ssd_300_vgg/conv10_1/weights/Initializer/random_uniform/min*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*
T0*(
_output_shapes
:??
?
ssd_300_vgg/conv10_1/weights
VariableV2*
shared_name *
dtype0*
	container *(
_output_shapes
:??*
shape:??*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights
?
#ssd_300_vgg/conv10_1/weights/AssignAssignssd_300_vgg/conv10_1/weights7ssd_300_vgg/conv10_1/weights/Initializer/random_uniform*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*
use_locking(*
T0*
validate_shape(*(
_output_shapes
:??
?
!ssd_300_vgg/conv10_1/weights/readIdentityssd_300_vgg/conv10_1/weights*
T0*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*(
_output_shapes
:??
?
<ssd_300_vgg/conv10_1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
=ssd_300_vgg/conv10_1/kernel/Regularizer/l2_regularizer/L2LossL2Loss!ssd_300_vgg/conv10_1/weights/read*
T0*
_output_shapes
: 
?
6ssd_300_vgg/conv10_1/kernel/Regularizer/l2_regularizerMul<ssd_300_vgg/conv10_1/kernel/Regularizer/l2_regularizer/scale=ssd_300_vgg/conv10_1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
-ssd_300_vgg/conv10_1/biases/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*.
_class$
" loc:@ssd_300_vgg/conv10_1/biases*
dtype0
?
ssd_300_vgg/conv10_1/biases
VariableV2*
_output_shapes	
:?*.
_class$
" loc:@ssd_300_vgg/conv10_1/biases*
	container *
shared_name *
dtype0*
shape:?
?
"ssd_300_vgg/conv10_1/biases/AssignAssignssd_300_vgg/conv10_1/biases-ssd_300_vgg/conv10_1/biases/Initializer/zeros*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(*.
_class$
" loc:@ssd_300_vgg/conv10_1/biases
?
 ssd_300_vgg/conv10_1/biases/readIdentityssd_300_vgg/conv10_1/biases*.
_class$
" loc:@ssd_300_vgg/conv10_1/biases*
T0*
_output_shapes	
:?
s
"ssd_300_vgg/conv10_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
ssd_300_vgg/conv10_1/Conv2DConv2Dssd_300_vgg/Relu_4!ssd_300_vgg/conv10_1/weights/read*
data_formatNHWC*
strides
*
T0*
	dilations
*
paddingSAME*'
_output_shapes
:?*
use_cudnn_on_gpu(
?
ssd_300_vgg/conv10_1/BiasAddBiasAddssd_300_vgg/conv10_1/Conv2D ssd_300_vgg/conv10_1/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:?
j
ssd_300_vgg/Relu_5Relussd_300_vgg/conv10_1/BiasAdd*'
_output_shapes
:?*
T0
?
ssd_300_vgg/Pad_2/paddingsConst*
dtype0*9
value0B."                             *
_output_shapes

:
?
ssd_300_vgg/Pad_2Padssd_300_vgg/Relu_5ssd_300_vgg/Pad_2/paddings*
T0*
	Tpaddings0*'
_output_shapes
:?
?
=ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/shapeConst*/
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   
?
;ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/minConst*/
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*
dtype0*
valueB
 *?Q?*
_output_shapes
: 
?
;ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/maxConst*
valueB
 *?Q=*
_output_shapes
: */
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*
dtype0
?
Essd_300_vgg/conv10_2/weights/Initializer/random_uniform/RandomUniformRandomUniform=ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/shape*

seed *
seed2 */
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*(
_output_shapes
:??*
T0*
dtype0
?
;ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/subSub;ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/max;ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/min*/
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*
_output_shapes
: *
T0
?
;ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/mulMulEssd_300_vgg/conv10_2/weights/Initializer/random_uniform/RandomUniform;ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*/
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*
T0
?
7ssd_300_vgg/conv10_2/weights/Initializer/random_uniformAdd;ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/mul;ssd_300_vgg/conv10_2/weights/Initializer/random_uniform/min*(
_output_shapes
:??*/
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*
T0
?
ssd_300_vgg/conv10_2/weights
VariableV2*
shape:??*
dtype0*
	container */
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*
shared_name *(
_output_shapes
:??
?
#ssd_300_vgg/conv10_2/weights/AssignAssignssd_300_vgg/conv10_2/weights7ssd_300_vgg/conv10_2/weights/Initializer/random_uniform*
T0*
validate_shape(*/
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*(
_output_shapes
:??*
use_locking(
?
!ssd_300_vgg/conv10_2/weights/readIdentityssd_300_vgg/conv10_2/weights*/
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*
T0*(
_output_shapes
:??
?
<ssd_300_vgg/conv10_2/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *o:
?
=ssd_300_vgg/conv10_2/kernel/Regularizer/l2_regularizer/L2LossL2Loss!ssd_300_vgg/conv10_2/weights/read*
_output_shapes
: *
T0
?
6ssd_300_vgg/conv10_2/kernel/Regularizer/l2_regularizerMul<ssd_300_vgg/conv10_2/kernel/Regularizer/l2_regularizer/scale=ssd_300_vgg/conv10_2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
-ssd_300_vgg/conv10_2/biases/Initializer/zerosConst*
dtype0*.
_class$
" loc:@ssd_300_vgg/conv10_2/biases*
_output_shapes	
:?*
valueB?*    
?
ssd_300_vgg/conv10_2/biases
VariableV2*
shape:?*.
_class$
" loc:@ssd_300_vgg/conv10_2/biases*
shared_name *
dtype0*
_output_shapes	
:?*
	container 
?
"ssd_300_vgg/conv10_2/biases/AssignAssignssd_300_vgg/conv10_2/biases-ssd_300_vgg/conv10_2/biases/Initializer/zeros*
_output_shapes	
:?*
T0*.
_class$
" loc:@ssd_300_vgg/conv10_2/biases*
use_locking(*
validate_shape(
?
 ssd_300_vgg/conv10_2/biases/readIdentityssd_300_vgg/conv10_2/biases*
T0*
_output_shapes	
:?*.
_class$
" loc:@ssd_300_vgg/conv10_2/biases
s
"ssd_300_vgg/conv10_2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
ssd_300_vgg/conv10_2/Conv2DConv2Dssd_300_vgg/Pad_2!ssd_300_vgg/conv10_2/weights/read*'
_output_shapes
:?*
use_cudnn_on_gpu(*
data_formatNHWC*
paddingVALID*
strides
*
T0*
	dilations

?
ssd_300_vgg/conv10_2/BiasAddBiasAddssd_300_vgg/conv10_2/Conv2D ssd_300_vgg/conv10_2/biases/read*'
_output_shapes
:?*
data_formatNHWC*
T0
j
ssd_300_vgg/Relu_6Relussd_300_vgg/conv10_2/BiasAdd*
T0*'
_output_shapes
:?
r
!ssd_300_vgg/gag/reduction_indicesConst*
valueB"      *
_output_shapes
:*
dtype0
?
ssd_300_vgg/gagMeanssd_300_vgg/Relu_6!ssd_300_vgg/gag/reduction_indices*

Tidx0*'
_output_shapes
:?*
T0*
	keep_dims(
t
2ssd_300_vgg/block4_box/L2Normalization/range/startConst*
_output_shapes
: *
value	B :*
dtype0
t
2ssd_300_vgg/block4_box/L2Normalization/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
t
2ssd_300_vgg/block4_box/L2Normalization/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
?
,ssd_300_vgg/block4_box/L2Normalization/rangeRange2ssd_300_vgg/block4_box/L2Normalization/range/start2ssd_300_vgg/block4_box/L2Normalization/range/limit2ssd_300_vgg/block4_box/L2Normalization/range/delta*

Tidx0*
_output_shapes
:
?
:ssd_300_vgg/block4_box/L2Normalization/l2_normalize/SquareSquaressd_300_vgg/resblock1_1/add*
T0*'
_output_shapes
:?
?
7ssd_300_vgg/block4_box/L2Normalization/l2_normalize/SumSum:ssd_300_vgg/block4_box/L2Normalization/l2_normalize/Square,ssd_300_vgg/block4_box/L2Normalization/range*
	keep_dims(*

Tidx0*&
_output_shapes
:*
T0
?
=ssd_300_vgg/block4_box/L2Normalization/l2_normalize/Maximum/yConst*
valueB
 *̼?+*
_output_shapes
: *
dtype0
?
;ssd_300_vgg/block4_box/L2Normalization/l2_normalize/MaximumMaximum7ssd_300_vgg/block4_box/L2Normalization/l2_normalize/Sum=ssd_300_vgg/block4_box/L2Normalization/l2_normalize/Maximum/y*
T0*&
_output_shapes
:
?
9ssd_300_vgg/block4_box/L2Normalization/l2_normalize/RsqrtRsqrt;ssd_300_vgg/block4_box/L2Normalization/l2_normalize/Maximum*
T0*&
_output_shapes
:
?
3ssd_300_vgg/block4_box/L2Normalization/l2_normalizeMulssd_300_vgg/resblock1_1/add9ssd_300_vgg/block4_box/L2Normalization/l2_normalize/Rsqrt*'
_output_shapes
:?*
T0
?
=ssd_300_vgg/block4_box/L2Normalization/gamma/Initializer/onesConst*
dtype0*?
_class5
31loc:@ssd_300_vgg/block4_box/L2Normalization/gamma*
_output_shapes	
:?*
valueB?*  ??
?
,ssd_300_vgg/block4_box/L2Normalization/gamma
VariableV2*
	container *
shape:?*
_output_shapes	
:?*
shared_name *
dtype0*?
_class5
31loc:@ssd_300_vgg/block4_box/L2Normalization/gamma
?
3ssd_300_vgg/block4_box/L2Normalization/gamma/AssignAssign,ssd_300_vgg/block4_box/L2Normalization/gamma=ssd_300_vgg/block4_box/L2Normalization/gamma/Initializer/ones*
use_locking(*
T0*?
_class5
31loc:@ssd_300_vgg/block4_box/L2Normalization/gamma*
validate_shape(*
_output_shapes	
:?
?
1ssd_300_vgg/block4_box/L2Normalization/gamma/readIdentity,ssd_300_vgg/block4_box/L2Normalization/gamma*?
_class5
31loc:@ssd_300_vgg/block4_box/L2Normalization/gamma*
T0*
_output_shapes	
:?
?
*ssd_300_vgg/block4_box/L2Normalization/MulMul3ssd_300_vgg/block4_box/L2Normalization/l2_normalize1ssd_300_vgg/block4_box/L2Normalization/gamma/read*
T0*'
_output_shapes
:?
?
Hssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/shapeConst*%
valueB"      ?      *:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
dtype0*
_output_shapes
:
?
Fssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/minConst*
valueB
 *HY??*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
dtype0
?
Fssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
valueB
 *HY?=
?
Pssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/shape*'
_output_shapes
:?*
dtype0*

seed *
T0*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
seed2 
?
Fssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/subSubFssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/maxFssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/min*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
T0
?
Fssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/mulMulPssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/sub*
T0*'
_output_shapes
:?*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights
?
Bssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniformAddFssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/mulFssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform/min*'
_output_shapes
:?*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
T0
?
'ssd_300_vgg/block4_box/conv_loc/weights
VariableV2*
shape:?*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
	container *'
_output_shapes
:?*
shared_name *
dtype0
?
.ssd_300_vgg/block4_box/conv_loc/weights/AssignAssign'ssd_300_vgg/block4_box/conv_loc/weightsBssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
T0*'
_output_shapes
:?*
use_locking(*
validate_shape(
?
,ssd_300_vgg/block4_box/conv_loc/weights/readIdentity'ssd_300_vgg/block4_box/conv_loc/weights*'
_output_shapes
:?*
T0*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights
?
Gssd_300_vgg/block4_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Hssd_300_vgg/block4_box/conv_loc/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/block4_box/conv_loc/weights/read*
T0*
_output_shapes
: 
?
Assd_300_vgg/block4_box/conv_loc/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/block4_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/block4_box/conv_loc/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
8ssd_300_vgg/block4_box/conv_loc/biases/Initializer/zerosConst*
valueB*    *9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_loc/biases*
dtype0*
_output_shapes
:
?
&ssd_300_vgg/block4_box/conv_loc/biases
VariableV2*
dtype0*
_output_shapes
:*
shape:*
shared_name *9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_loc/biases*
	container 
?
-ssd_300_vgg/block4_box/conv_loc/biases/AssignAssign&ssd_300_vgg/block4_box/conv_loc/biases8ssd_300_vgg/block4_box/conv_loc/biases/Initializer/zeros*9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_loc/biases*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
?
+ssd_300_vgg/block4_box/conv_loc/biases/readIdentity&ssd_300_vgg/block4_box/conv_loc/biases*
_output_shapes
:*
T0*9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_loc/biases
~
-ssd_300_vgg/block4_box/conv_loc/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
&ssd_300_vgg/block4_box/conv_loc/Conv2DConv2D*ssd_300_vgg/block4_box/L2Normalization/Mul,ssd_300_vgg/block4_box/conv_loc/weights/read*
data_formatNHWC*
use_cudnn_on_gpu(*&
_output_shapes
:*
	dilations
*
T0*
paddingSAME*
strides

?
'ssd_300_vgg/block4_box/conv_loc/BiasAddBiasAdd&ssd_300_vgg/block4_box/conv_loc/Conv2D+ssd_300_vgg/block4_box/conv_loc/biases/read*
data_formatNHWC*&
_output_shapes
:*
T0
?
$ssd_300_vgg/block4_box/Reshape/shapeConst*
_output_shapes
:*)
value B"               *
dtype0
?
ssd_300_vgg/block4_box/ReshapeReshape'ssd_300_vgg/block4_box/conv_loc/BiasAdd$ssd_300_vgg/block4_box/Reshape/shape*
T0*
Tshape0**
_output_shapes
:
?
Hssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights*
_output_shapes
:*%
valueB"      ?   T   *
dtype0
?
Fssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights*
valueB
 *9?e?
?
Fssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights*
valueB
 *9?e=*
dtype0
?
Pssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/shape*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights*'
_output_shapes
:?T*
seed2 *
dtype0*

seed *
T0
?
Fssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/subSubFssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/maxFssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/min*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights*
_output_shapes
: *
T0
?
Fssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/mulMulPssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/sub*
T0*'
_output_shapes
:?T*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights
?
Bssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniformAddFssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/mulFssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform/min*
T0*'
_output_shapes
:?T*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights
?
'ssd_300_vgg/block4_box/conv_cls/weights
VariableV2*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights*
shared_name *
shape:?T*'
_output_shapes
:?T*
dtype0*
	container 
?
.ssd_300_vgg/block4_box/conv_cls/weights/AssignAssign'ssd_300_vgg/block4_box/conv_cls/weightsBssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:?T
?
,ssd_300_vgg/block4_box/conv_cls/weights/readIdentity'ssd_300_vgg/block4_box/conv_cls/weights*
T0*'
_output_shapes
:?T*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights
?
Gssd_300_vgg/block4_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
?
Hssd_300_vgg/block4_box/conv_cls/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/block4_box/conv_cls/weights/read*
T0*
_output_shapes
: 
?
Assd_300_vgg/block4_box/conv_cls/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/block4_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/block4_box/conv_cls/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
8ssd_300_vgg/block4_box/conv_cls/biases/Initializer/zerosConst*9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_cls/biases*
dtype0*
valueBT*    *
_output_shapes
:T
?
&ssd_300_vgg/block4_box/conv_cls/biases
VariableV2*
_output_shapes
:T*
shape:T*
shared_name *
dtype0*
	container *9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_cls/biases
?
-ssd_300_vgg/block4_box/conv_cls/biases/AssignAssign&ssd_300_vgg/block4_box/conv_cls/biases8ssd_300_vgg/block4_box/conv_cls/biases/Initializer/zeros*
use_locking(*
_output_shapes
:T*
validate_shape(*
T0*9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_cls/biases
?
+ssd_300_vgg/block4_box/conv_cls/biases/readIdentity&ssd_300_vgg/block4_box/conv_cls/biases*
T0*
_output_shapes
:T*9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_cls/biases
~
-ssd_300_vgg/block4_box/conv_cls/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
&ssd_300_vgg/block4_box/conv_cls/Conv2DConv2D*ssd_300_vgg/block4_box/L2Normalization/Mul,ssd_300_vgg/block4_box/conv_cls/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:T*
	dilations
*
T0*
strides

?
'ssd_300_vgg/block4_box/conv_cls/BiasAddBiasAdd&ssd_300_vgg/block4_box/conv_cls/Conv2D+ssd_300_vgg/block4_box/conv_cls/biases/read*&
_output_shapes
:T*
data_formatNHWC*
T0
?
&ssd_300_vgg/block4_box/Reshape_1/shapeConst*
dtype0*)
value B"               *
_output_shapes
:
?
 ssd_300_vgg/block4_box/Reshape_1Reshape'ssd_300_vgg/block4_box/conv_cls/BiasAdd&ssd_300_vgg/block4_box/Reshape_1/shape*
Tshape0**
_output_shapes
:*
T0
r
!ssd_300_vgg/softmax/Reshape/shapeConst*
_output_shapes
:*
valueB"????   *
dtype0
?
ssd_300_vgg/softmax/ReshapeReshape ssd_300_vgg/block4_box/Reshape_1!ssd_300_vgg/softmax/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	?
m
ssd_300_vgg/softmax/SoftmaxSoftmaxssd_300_vgg/softmax/Reshape*
_output_shapes
:	?*
T0
v
ssd_300_vgg/softmax/ShapeConst*
dtype0*)
value B"               *
_output_shapes
:
?
ssd_300_vgg/softmax/Reshape_1Reshapessd_300_vgg/softmax/Softmaxssd_300_vgg/softmax/Shape*
T0*
Tshape0**
_output_shapes
:
?
Hssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"            *:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights
?
Fssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/minConst*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights*
_output_shapes
: *
valueB
 *E?G?*
dtype0
?
Fssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights*
dtype0*
valueB
 *E?G=
?
Pssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/shape*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights*'
_output_shapes
:?*
seed2 *
dtype0*
T0*

seed 
?
Fssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/subSubFssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/maxFssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/min*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights*
T0*
_output_shapes
: 
?
Fssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/mulMulPssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/sub*
T0*'
_output_shapes
:?*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights
?
Bssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniformAddFssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/mulFssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights*'
_output_shapes
:?
?
'ssd_300_vgg/block7_box/conv_loc/weights
VariableV2*
	container *'
_output_shapes
:?*
shared_name *
shape:?*
dtype0*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights
?
.ssd_300_vgg/block7_box/conv_loc/weights/AssignAssign'ssd_300_vgg/block7_box/conv_loc/weightsBssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:?
?
,ssd_300_vgg/block7_box/conv_loc/weights/readIdentity'ssd_300_vgg/block7_box/conv_loc/weights*'
_output_shapes
:?*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights*
T0
?
Gssd_300_vgg/block7_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
?
Hssd_300_vgg/block7_box/conv_loc/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/block7_box/conv_loc/weights/read*
T0*
_output_shapes
: 
?
Assd_300_vgg/block7_box/conv_loc/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/block7_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/block7_box/conv_loc/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
8ssd_300_vgg/block7_box/conv_loc/biases/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_loc/biases
?
&ssd_300_vgg/block7_box/conv_loc/biases
VariableV2*9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_loc/biases*
_output_shapes
:*
dtype0*
shape:*
shared_name *
	container 
?
-ssd_300_vgg/block7_box/conv_loc/biases/AssignAssign&ssd_300_vgg/block7_box/conv_loc/biases8ssd_300_vgg/block7_box/conv_loc/biases/Initializer/zeros*9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_loc/biases*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
?
+ssd_300_vgg/block7_box/conv_loc/biases/readIdentity&ssd_300_vgg/block7_box/conv_loc/biases*9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_loc/biases*
_output_shapes
:*
T0
~
-ssd_300_vgg/block7_box/conv_loc/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
&ssd_300_vgg/block7_box/conv_loc/Conv2DConv2Dssd_300_vgg/resblock2_1/add,ssd_300_vgg/block7_box/conv_loc/weights/read*
T0*
strides
*
data_formatNHWC*
	dilations
*
use_cudnn_on_gpu(*&
_output_shapes
:*
paddingSAME
?
'ssd_300_vgg/block7_box/conv_loc/BiasAddBiasAdd&ssd_300_vgg/block7_box/conv_loc/Conv2D+ssd_300_vgg/block7_box/conv_loc/biases/read*&
_output_shapes
:*
data_formatNHWC*
T0
?
$ssd_300_vgg/block7_box/Reshape/shapeConst*)
value B"               *
dtype0*
_output_shapes
:
?
ssd_300_vgg/block7_box/ReshapeReshape'ssd_300_vgg/block7_box/conv_loc/BiasAdd$ssd_300_vgg/block7_box/Reshape/shape*
T0*
Tshape0**
_output_shapes
:
?
Hssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"         ~   *
dtype0*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights
?
Fssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/minConst*
valueB
 *?+?*
dtype0*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights
?
Fssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/maxConst*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights*
_output_shapes
: *
dtype0*
valueB
 *?+=
?
Pssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/shape*
seed2 *'
_output_shapes
:?~*
T0*

seed *:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights*
dtype0
?
Fssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/subSubFssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/maxFssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights
?
Fssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/mulMulPssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/sub*
T0*'
_output_shapes
:?~*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights
?
Bssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniformAddFssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/mulFssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform/min*'
_output_shapes
:?~*
T0*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights
?
'ssd_300_vgg/block7_box/conv_cls/weights
VariableV2*'
_output_shapes
:?~*
shape:?~*
dtype0*
	container *:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights*
shared_name 
?
.ssd_300_vgg/block7_box/conv_cls/weights/AssignAssign'ssd_300_vgg/block7_box/conv_cls/weightsBssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform*
use_locking(*
validate_shape(*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights*'
_output_shapes
:?~*
T0
?
,ssd_300_vgg/block7_box/conv_cls/weights/readIdentity'ssd_300_vgg/block7_box/conv_cls/weights*'
_output_shapes
:?~*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights*
T0
?
Gssd_300_vgg/block7_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:
?
Hssd_300_vgg/block7_box/conv_cls/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/block7_box/conv_cls/weights/read*
T0*
_output_shapes
: 
?
Assd_300_vgg/block7_box/conv_cls/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/block7_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/block7_box/conv_cls/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
8ssd_300_vgg/block7_box/conv_cls/biases/Initializer/zerosConst*
valueB~*    *9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_cls/biases*
_output_shapes
:~*
dtype0
?
&ssd_300_vgg/block7_box/conv_cls/biases
VariableV2*
	container *9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_cls/biases*
shape:~*
dtype0*
shared_name *
_output_shapes
:~
?
-ssd_300_vgg/block7_box/conv_cls/biases/AssignAssign&ssd_300_vgg/block7_box/conv_cls/biases8ssd_300_vgg/block7_box/conv_cls/biases/Initializer/zeros*
_output_shapes
:~*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_cls/biases*
validate_shape(*
T0
?
+ssd_300_vgg/block7_box/conv_cls/biases/readIdentity&ssd_300_vgg/block7_box/conv_cls/biases*
_output_shapes
:~*
T0*9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_cls/biases
~
-ssd_300_vgg/block7_box/conv_cls/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
&ssd_300_vgg/block7_box/conv_cls/Conv2DConv2Dssd_300_vgg/resblock2_1/add,ssd_300_vgg/block7_box/conv_cls/weights/read*
paddingSAME*&
_output_shapes
:~*
	dilations
*
data_formatNHWC*
T0*
use_cudnn_on_gpu(*
strides

?
'ssd_300_vgg/block7_box/conv_cls/BiasAddBiasAdd&ssd_300_vgg/block7_box/conv_cls/Conv2D+ssd_300_vgg/block7_box/conv_cls/biases/read*&
_output_shapes
:~*
T0*
data_formatNHWC
?
&ssd_300_vgg/block7_box/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"               
?
 ssd_300_vgg/block7_box/Reshape_1Reshape'ssd_300_vgg/block7_box/conv_cls/BiasAdd&ssd_300_vgg/block7_box/Reshape_1/shape**
_output_shapes
:*
T0*
Tshape0
t
#ssd_300_vgg/softmax_1/Reshape/shapeConst*
valueB"????   *
_output_shapes
:*
dtype0
?
ssd_300_vgg/softmax_1/ReshapeReshape ssd_300_vgg/block7_box/Reshape_1#ssd_300_vgg/softmax_1/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	?	
q
ssd_300_vgg/softmax_1/SoftmaxSoftmaxssd_300_vgg/softmax_1/Reshape*
_output_shapes
:	?	*
T0
x
ssd_300_vgg/softmax_1/ShapeConst*)
value B"               *
_output_shapes
:*
dtype0
?
ssd_300_vgg/softmax_1/Reshape_1Reshapessd_300_vgg/softmax_1/Softmaxssd_300_vgg/softmax_1/Shape**
_output_shapes
:*
T0*
Tshape0
?
Hssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights*
dtype0*%
valueB"      ?      *
_output_shapes
:
?
Fssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/minConst*
valueB
 *ҡ??*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights*
dtype0
?
Fssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *ҡ?=*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights
?
Pssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/shape*
dtype0*

seed *
T0*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights*
seed2 *'
_output_shapes
:?
?
Fssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/subSubFssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/maxFssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/min*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights*
T0*
_output_shapes
: 
?
Fssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/mulMulPssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights*'
_output_shapes
:?
?
Bssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniformAddFssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/mulFssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform/min*'
_output_shapes
:?*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights*
T0
?
'ssd_300_vgg/block8_box/conv_loc/weights
VariableV2*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights*
shared_name *'
_output_shapes
:?*
	container *
shape:?*
dtype0
?
.ssd_300_vgg/block8_box/conv_loc/weights/AssignAssign'ssd_300_vgg/block8_box/conv_loc/weightsBssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*'
_output_shapes
:?*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights
?
,ssd_300_vgg/block8_box/conv_loc/weights/readIdentity'ssd_300_vgg/block8_box/conv_loc/weights*
T0*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights*'
_output_shapes
:?
?
Gssd_300_vgg/block8_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
?
Hssd_300_vgg/block8_box/conv_loc/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/block8_box/conv_loc/weights/read*
_output_shapes
: *
T0
?
Assd_300_vgg/block8_box/conv_loc/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/block8_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/block8_box/conv_loc/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
8ssd_300_vgg/block8_box/conv_loc/biases/Initializer/zerosConst*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_loc/biases*
dtype0*
_output_shapes
:*
valueB*    
?
&ssd_300_vgg/block8_box/conv_loc/biases
VariableV2*
shape:*
	container *
_output_shapes
:*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_loc/biases*
shared_name *
dtype0
?
-ssd_300_vgg/block8_box/conv_loc/biases/AssignAssign&ssd_300_vgg/block8_box/conv_loc/biases8ssd_300_vgg/block8_box/conv_loc/biases/Initializer/zeros*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_loc/biases
?
+ssd_300_vgg/block8_box/conv_loc/biases/readIdentity&ssd_300_vgg/block8_box/conv_loc/biases*
T0*
_output_shapes
:*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_loc/biases
~
-ssd_300_vgg/block8_box/conv_loc/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
&ssd_300_vgg/block8_box/conv_loc/Conv2DConv2Dssd_300_vgg/Relu_2,ssd_300_vgg/block8_box/conv_loc/weights/read*
use_cudnn_on_gpu(*
	dilations
*
strides
*
T0*&
_output_shapes
:*
paddingSAME*
data_formatNHWC
?
'ssd_300_vgg/block8_box/conv_loc/BiasAddBiasAdd&ssd_300_vgg/block8_box/conv_loc/Conv2D+ssd_300_vgg/block8_box/conv_loc/biases/read*
data_formatNHWC*
T0*&
_output_shapes
:
?
$ssd_300_vgg/block8_box/Reshape/shapeConst*)
value B"               *
dtype0*
_output_shapes
:
?
ssd_300_vgg/block8_box/ReshapeReshape'ssd_300_vgg/block8_box/conv_loc/BiasAdd$ssd_300_vgg/block8_box/Reshape/shape**
_output_shapes
:*
T0*
Tshape0
?
Hssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/shapeConst*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights*%
valueB"      ?   ~   *
dtype0*
_output_shapes
:
?
Fssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/minConst*
valueB
 *-?Q?*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights*
_output_shapes
: *
dtype0
?
Fssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *-?Q=*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights*
dtype0
?
Pssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*'
_output_shapes
:?~*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights
?
Fssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/subSubFssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/maxFssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/min*
T0*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights*
_output_shapes
: 
?
Fssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/mulMulPssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/sub*'
_output_shapes
:?~*
T0*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights
?
Bssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniformAddFssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/mulFssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform/min*'
_output_shapes
:?~*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights*
T0
?
'ssd_300_vgg/block8_box/conv_cls/weights
VariableV2*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights*
shared_name *
	container *
dtype0*
shape:?~*'
_output_shapes
:?~
?
.ssd_300_vgg/block8_box/conv_cls/weights/AssignAssign'ssd_300_vgg/block8_box/conv_cls/weightsBssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform*
use_locking(*
validate_shape(*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights*
T0*'
_output_shapes
:?~
?
,ssd_300_vgg/block8_box/conv_cls/weights/readIdentity'ssd_300_vgg/block8_box/conv_cls/weights*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights*
T0*'
_output_shapes
:?~
?
Gssd_300_vgg/block8_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Hssd_300_vgg/block8_box/conv_cls/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/block8_box/conv_cls/weights/read*
T0*
_output_shapes
: 
?
Assd_300_vgg/block8_box/conv_cls/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/block8_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/block8_box/conv_cls/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
8ssd_300_vgg/block8_box/conv_cls/biases/Initializer/zerosConst*
dtype0*
_output_shapes
:~*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_cls/biases*
valueB~*    
?
&ssd_300_vgg/block8_box/conv_cls/biases
VariableV2*
	container *
shared_name *
_output_shapes
:~*
dtype0*
shape:~*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_cls/biases
?
-ssd_300_vgg/block8_box/conv_cls/biases/AssignAssign&ssd_300_vgg/block8_box/conv_cls/biases8ssd_300_vgg/block8_box/conv_cls/biases/Initializer/zeros*
_output_shapes
:~*
use_locking(*
T0*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_cls/biases*
validate_shape(
?
+ssd_300_vgg/block8_box/conv_cls/biases/readIdentity&ssd_300_vgg/block8_box/conv_cls/biases*
_output_shapes
:~*
T0*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_cls/biases
~
-ssd_300_vgg/block8_box/conv_cls/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
&ssd_300_vgg/block8_box/conv_cls/Conv2DConv2Dssd_300_vgg/Relu_2,ssd_300_vgg/block8_box/conv_cls/weights/read*
	dilations
*
use_cudnn_on_gpu(*&
_output_shapes
:~*
T0*
strides
*
data_formatNHWC*
paddingSAME
?
'ssd_300_vgg/block8_box/conv_cls/BiasAddBiasAdd&ssd_300_vgg/block8_box/conv_cls/Conv2D+ssd_300_vgg/block8_box/conv_cls/biases/read*
T0*&
_output_shapes
:~*
data_formatNHWC
?
&ssd_300_vgg/block8_box/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*)
value B"               
?
 ssd_300_vgg/block8_box/Reshape_1Reshape'ssd_300_vgg/block8_box/conv_cls/BiasAdd&ssd_300_vgg/block8_box/Reshape_1/shape*
T0**
_output_shapes
:*
Tshape0
t
#ssd_300_vgg/softmax_2/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   
?
ssd_300_vgg/softmax_2/ReshapeReshape ssd_300_vgg/block8_box/Reshape_1#ssd_300_vgg/softmax_2/Reshape/shape*
T0*
_output_shapes
:	?*
Tshape0
q
ssd_300_vgg/softmax_2/SoftmaxSoftmaxssd_300_vgg/softmax_2/Reshape*
_output_shapes
:	?*
T0
x
ssd_300_vgg/softmax_2/ShapeConst*)
value B"               *
dtype0*
_output_shapes
:
?
ssd_300_vgg/softmax_2/Reshape_1Reshapessd_300_vgg/softmax_2/Softmaxssd_300_vgg/softmax_2/Shape*
Tshape0*
T0**
_output_shapes
:
?
Hssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*%
valueB"      ?      *
dtype0
?
Fssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/minConst*
valueB
 *ҡ??*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*
dtype0
?
Fssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/maxConst*
valueB
 *ҡ?=*
dtype0*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*
_output_shapes
: 
?
Pssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/shape*
dtype0*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*
seed2 *

seed *'
_output_shapes
:?*
T0
?
Fssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/subSubFssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/maxFssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights
?
Fssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/mulMulPssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/sub*
T0*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*'
_output_shapes
:?
?
Bssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniformAddFssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/mulFssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform/min*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*
T0*'
_output_shapes
:?
?
'ssd_300_vgg/block9_box/conv_loc/weights
VariableV2*
shape:?*
dtype0*'
_output_shapes
:?*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*
shared_name *
	container 
?
.ssd_300_vgg/block9_box/conv_loc/weights/AssignAssign'ssd_300_vgg/block9_box/conv_loc/weightsBssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform*
use_locking(*'
_output_shapes
:?*
T0*
validate_shape(*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights
?
,ssd_300_vgg/block9_box/conv_loc/weights/readIdentity'ssd_300_vgg/block9_box/conv_loc/weights*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*
T0*'
_output_shapes
:?
?
Gssd_300_vgg/block9_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
_output_shapes
: *
dtype0
?
Hssd_300_vgg/block9_box/conv_loc/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/block9_box/conv_loc/weights/read*
T0*
_output_shapes
: 
?
Assd_300_vgg/block9_box/conv_loc/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/block9_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/block9_box/conv_loc/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
8ssd_300_vgg/block9_box/conv_loc/biases/Initializer/zerosConst*
_output_shapes
:*
valueB*    *9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_loc/biases*
dtype0
?
&ssd_300_vgg/block9_box/conv_loc/biases
VariableV2*
_output_shapes
:*
shared_name *
shape:*
	container *9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_loc/biases*
dtype0
?
-ssd_300_vgg/block9_box/conv_loc/biases/AssignAssign&ssd_300_vgg/block9_box/conv_loc/biases8ssd_300_vgg/block9_box/conv_loc/biases/Initializer/zeros*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_loc/biases
?
+ssd_300_vgg/block9_box/conv_loc/biases/readIdentity&ssd_300_vgg/block9_box/conv_loc/biases*
_output_shapes
:*
T0*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_loc/biases
~
-ssd_300_vgg/block9_box/conv_loc/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
&ssd_300_vgg/block9_box/conv_loc/Conv2DConv2Dssd_300_vgg/Relu_4,ssd_300_vgg/block9_box/conv_loc/weights/read*
data_formatNHWC*
use_cudnn_on_gpu(*
strides
*
paddingSAME*&
_output_shapes
:*
	dilations
*
T0
?
'ssd_300_vgg/block9_box/conv_loc/BiasAddBiasAdd&ssd_300_vgg/block9_box/conv_loc/Conv2D+ssd_300_vgg/block9_box/conv_loc/biases/read*
data_formatNHWC*&
_output_shapes
:*
T0
?
$ssd_300_vgg/block9_box/Reshape/shapeConst*
dtype0*)
value B"               *
_output_shapes
:
?
ssd_300_vgg/block9_box/ReshapeReshape'ssd_300_vgg/block9_box/conv_loc/BiasAdd$ssd_300_vgg/block9_box/Reshape/shape*
T0*
Tshape0**
_output_shapes
:
?
Hssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      ?   ~   *:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*
_output_shapes
:
?
Fssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/minConst*
valueB
 *-?Q?*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*
dtype0*
_output_shapes
: 
?
Fssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/maxConst*
valueB
 *-?Q=*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*
dtype0*
_output_shapes
: 
?
Pssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/RandomUniformRandomUniformHssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/shape*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*
seed2 *
T0*'
_output_shapes
:?~*

seed *
dtype0
?
Fssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/subSubFssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/maxFssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/min*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*
T0*
_output_shapes
: 
?
Fssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/mulMulPssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/RandomUniformFssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/sub*
T0*'
_output_shapes
:?~*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights
?
Bssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniformAddFssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/mulFssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform/min*'
_output_shapes
:?~*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*
T0
?
'ssd_300_vgg/block9_box/conv_cls/weights
VariableV2*
shape:?~*'
_output_shapes
:?~*
dtype0*
	container *:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*
shared_name 
?
.ssd_300_vgg/block9_box/conv_cls/weights/AssignAssign'ssd_300_vgg/block9_box/conv_cls/weightsBssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform*'
_output_shapes
:?~*
use_locking(*
validate_shape(*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*
T0
?
,ssd_300_vgg/block9_box/conv_cls/weights/readIdentity'ssd_300_vgg/block9_box/conv_cls/weights*'
_output_shapes
:?~*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*
T0
?
Gssd_300_vgg/block9_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:
?
Hssd_300_vgg/block9_box/conv_cls/kernel/Regularizer/l2_regularizer/L2LossL2Loss,ssd_300_vgg/block9_box/conv_cls/weights/read*
_output_shapes
: *
T0
?
Assd_300_vgg/block9_box/conv_cls/kernel/Regularizer/l2_regularizerMulGssd_300_vgg/block9_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleHssd_300_vgg/block9_box/conv_cls/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
8ssd_300_vgg/block9_box/conv_cls/biases/Initializer/zerosConst*
_output_shapes
:~*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_cls/biases*
dtype0*
valueB~*    
?
&ssd_300_vgg/block9_box/conv_cls/biases
VariableV2*
shared_name *
	container *
dtype0*
_output_shapes
:~*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_cls/biases*
shape:~
?
-ssd_300_vgg/block9_box/conv_cls/biases/AssignAssign&ssd_300_vgg/block9_box/conv_cls/biases8ssd_300_vgg/block9_box/conv_cls/biases/Initializer/zeros*
_output_shapes
:~*
T0*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_cls/biases*
validate_shape(
?
+ssd_300_vgg/block9_box/conv_cls/biases/readIdentity&ssd_300_vgg/block9_box/conv_cls/biases*
T0*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_cls/biases*
_output_shapes
:~
~
-ssd_300_vgg/block9_box/conv_cls/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
&ssd_300_vgg/block9_box/conv_cls/Conv2DConv2Dssd_300_vgg/Relu_4,ssd_300_vgg/block9_box/conv_cls/weights/read*&
_output_shapes
:~*
use_cudnn_on_gpu(*
paddingSAME*
T0*
strides
*
data_formatNHWC*
	dilations

?
'ssd_300_vgg/block9_box/conv_cls/BiasAddBiasAdd&ssd_300_vgg/block9_box/conv_cls/Conv2D+ssd_300_vgg/block9_box/conv_cls/biases/read*&
_output_shapes
:~*
data_formatNHWC*
T0
?
&ssd_300_vgg/block9_box/Reshape_1/shapeConst*
_output_shapes
:*)
value B"               *
dtype0
?
 ssd_300_vgg/block9_box/Reshape_1Reshape'ssd_300_vgg/block9_box/conv_cls/BiasAdd&ssd_300_vgg/block9_box/Reshape_1/shape*
Tshape0**
_output_shapes
:*
T0
t
#ssd_300_vgg/softmax_3/Reshape/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
ssd_300_vgg/softmax_3/ReshapeReshape ssd_300_vgg/block9_box/Reshape_1#ssd_300_vgg/softmax_3/Reshape/shape*
T0*
Tshape0*
_output_shapes

:`
p
ssd_300_vgg/softmax_3/SoftmaxSoftmaxssd_300_vgg/softmax_3/Reshape*
T0*
_output_shapes

:`
x
ssd_300_vgg/softmax_3/ShapeConst*
dtype0*)
value B"               *
_output_shapes
:
?
ssd_300_vgg/softmax_3/Reshape_1Reshapessd_300_vgg/softmax_3/Softmaxssd_300_vgg/softmax_3/Shape*
Tshape0*
T0**
_output_shapes
:
?
Issd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*%
valueB"      ?      
?
Gssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/minConst*
valueB
 *HY??*
dtype0*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*
_output_shapes
: 
?
Gssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *HY?=*
dtype0*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights
?
Qssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/RandomUniformRandomUniformIssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/shape*
T0*

seed *'
_output_shapes
:?*
seed2 *;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*
dtype0
?
Gssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/subSubGssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/maxGssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/min*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*
T0*
_output_shapes
: 
?
Gssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/mulMulQssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/RandomUniformGssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/sub*'
_output_shapes
:?*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*
T0
?
Cssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniformAddGssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/mulGssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform/min*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*'
_output_shapes
:?*
T0
?
(ssd_300_vgg/block10_box/conv_loc/weights
VariableV2*'
_output_shapes
:?*
	container *
shared_name *
dtype0*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*
shape:?
?
/ssd_300_vgg/block10_box/conv_loc/weights/AssignAssign(ssd_300_vgg/block10_box/conv_loc/weightsCssd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform*'
_output_shapes
:?*
validate_shape(*
use_locking(*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*
T0
?
-ssd_300_vgg/block10_box/conv_loc/weights/readIdentity(ssd_300_vgg/block10_box/conv_loc/weights*'
_output_shapes
:?*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*
T0
?
Hssd_300_vgg/block10_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Issd_300_vgg/block10_box/conv_loc/kernel/Regularizer/l2_regularizer/L2LossL2Loss-ssd_300_vgg/block10_box/conv_loc/weights/read*
_output_shapes
: *
T0
?
Bssd_300_vgg/block10_box/conv_loc/kernel/Regularizer/l2_regularizerMulHssd_300_vgg/block10_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleIssd_300_vgg/block10_box/conv_loc/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
?
9ssd_300_vgg/block10_box/conv_loc/biases/Initializer/zerosConst*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_loc/biases*
_output_shapes
:*
valueB*    *
dtype0
?
'ssd_300_vgg/block10_box/conv_loc/biases
VariableV2*
_output_shapes
:*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_loc/biases*
shape:*
shared_name *
dtype0*
	container 
?
.ssd_300_vgg/block10_box/conv_loc/biases/AssignAssign'ssd_300_vgg/block10_box/conv_loc/biases9ssd_300_vgg/block10_box/conv_loc/biases/Initializer/zeros*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_loc/biases*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
?
,ssd_300_vgg/block10_box/conv_loc/biases/readIdentity'ssd_300_vgg/block10_box/conv_loc/biases*
T0*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_loc/biases*
_output_shapes
:

.ssd_300_vgg/block10_box/conv_loc/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
'ssd_300_vgg/block10_box/conv_loc/Conv2DConv2Dssd_300_vgg/Relu_6-ssd_300_vgg/block10_box/conv_loc/weights/read*
T0*
use_cudnn_on_gpu(*
	dilations
*
data_formatNHWC*
strides
*
paddingSAME*&
_output_shapes
:
?
(ssd_300_vgg/block10_box/conv_loc/BiasAddBiasAdd'ssd_300_vgg/block10_box/conv_loc/Conv2D,ssd_300_vgg/block10_box/conv_loc/biases/read*
data_formatNHWC*
T0*&
_output_shapes
:
?
%ssd_300_vgg/block10_box/Reshape/shapeConst*
dtype0*
_output_shapes
:*)
value B"               
?
ssd_300_vgg/block10_box/ReshapeReshape(ssd_300_vgg/block10_box/conv_loc/BiasAdd%ssd_300_vgg/block10_box/Reshape/shape*
T0*
Tshape0**
_output_shapes
:
?
Issd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/shapeConst*%
valueB"      ?   T   *
_output_shapes
:*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights*
dtype0
?
Gssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/minConst*
valueB
 *9?e?*
_output_shapes
: *
dtype0*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights
?
Gssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/maxConst*
dtype0*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights*
valueB
 *9?e=*
_output_shapes
: 
?
Qssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/RandomUniformRandomUniformIssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/shape*
seed2 *
T0*

seed *
dtype0*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights*'
_output_shapes
:?T
?
Gssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/subSubGssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/maxGssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights
?
Gssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/mulMulQssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/RandomUniformGssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/sub*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights*'
_output_shapes
:?T*
T0
?
Cssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniformAddGssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/mulGssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights*'
_output_shapes
:?T
?
(ssd_300_vgg/block10_box/conv_cls/weights
VariableV2*
shared_name *;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights*
shape:?T*
dtype0*'
_output_shapes
:?T*
	container 
?
/ssd_300_vgg/block10_box/conv_cls/weights/AssignAssign(ssd_300_vgg/block10_box/conv_cls/weightsCssd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights*'
_output_shapes
:?T
?
-ssd_300_vgg/block10_box/conv_cls/weights/readIdentity(ssd_300_vgg/block10_box/conv_cls/weights*'
_output_shapes
:?T*
T0*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights
?
Hssd_300_vgg/block10_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
?
Issd_300_vgg/block10_box/conv_cls/kernel/Regularizer/l2_regularizer/L2LossL2Loss-ssd_300_vgg/block10_box/conv_cls/weights/read*
_output_shapes
: *
T0
?
Bssd_300_vgg/block10_box/conv_cls/kernel/Regularizer/l2_regularizerMulHssd_300_vgg/block10_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleIssd_300_vgg/block10_box/conv_cls/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
9ssd_300_vgg/block10_box/conv_cls/biases/Initializer/zerosConst*
dtype0*
_output_shapes
:T*
valueBT*    *:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_cls/biases
?
'ssd_300_vgg/block10_box/conv_cls/biases
VariableV2*
shared_name *
shape:T*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_cls/biases*
dtype0*
_output_shapes
:T*
	container 
?
.ssd_300_vgg/block10_box/conv_cls/biases/AssignAssign'ssd_300_vgg/block10_box/conv_cls/biases9ssd_300_vgg/block10_box/conv_cls/biases/Initializer/zeros*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_cls/biases*
validate_shape(*
use_locking(*
_output_shapes
:T*
T0
?
,ssd_300_vgg/block10_box/conv_cls/biases/readIdentity'ssd_300_vgg/block10_box/conv_cls/biases*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_cls/biases*
T0*
_output_shapes
:T

.ssd_300_vgg/block10_box/conv_cls/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
'ssd_300_vgg/block10_box/conv_cls/Conv2DConv2Dssd_300_vgg/Relu_6-ssd_300_vgg/block10_box/conv_cls/weights/read*&
_output_shapes
:T*
strides
*
	dilations
*
paddingSAME*
use_cudnn_on_gpu(*
T0*
data_formatNHWC
?
(ssd_300_vgg/block10_box/conv_cls/BiasAddBiasAdd'ssd_300_vgg/block10_box/conv_cls/Conv2D,ssd_300_vgg/block10_box/conv_cls/biases/read*
data_formatNHWC*
T0*&
_output_shapes
:T
?
'ssd_300_vgg/block10_box/Reshape_1/shapeConst*
_output_shapes
:*)
value B"               *
dtype0
?
!ssd_300_vgg/block10_box/Reshape_1Reshape(ssd_300_vgg/block10_box/conv_cls/BiasAdd'ssd_300_vgg/block10_box/Reshape_1/shape*
T0*
Tshape0**
_output_shapes
:
t
#ssd_300_vgg/softmax_4/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
ssd_300_vgg/softmax_4/ReshapeReshape!ssd_300_vgg/block10_box/Reshape_1#ssd_300_vgg/softmax_4/Reshape/shape*
T0*
_output_shapes

:*
Tshape0
p
ssd_300_vgg/softmax_4/SoftmaxSoftmaxssd_300_vgg/softmax_4/Reshape*
_output_shapes

:*
T0
x
ssd_300_vgg/softmax_4/ShapeConst*
_output_shapes
:*)
value B"               *
dtype0
?
ssd_300_vgg/softmax_4/Reshape_1Reshapessd_300_vgg/softmax_4/Softmaxssd_300_vgg/softmax_4/Shape**
_output_shapes
:*
T0*
Tshape0
?
Issd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/shapeConst*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights*
_output_shapes
:*%
valueB"      ?      *
dtype0
?
Gssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/minConst*
valueB
 *HY??*
dtype0*
_output_shapes
: *;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights
?
Gssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/maxConst*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights*
_output_shapes
: *
valueB
 *HY?=*
dtype0
?
Qssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/RandomUniformRandomUniformIssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/shape*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights*
T0*'
_output_shapes
:?*

seed *
seed2 *
dtype0
?
Gssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/subSubGssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/maxGssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights
?
Gssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/mulMulQssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/RandomUniformGssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/sub*
T0*'
_output_shapes
:?*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights
?
Cssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniformAddGssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/mulGssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights*'
_output_shapes
:?
?
(ssd_300_vgg/block11_box/conv_loc/weights
VariableV2*
shared_name *
shape:?*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights*
dtype0*
	container *'
_output_shapes
:?
?
/ssd_300_vgg/block11_box/conv_loc/weights/AssignAssign(ssd_300_vgg/block11_box/conv_loc/weightsCssd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform*
use_locking(*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights*
validate_shape(*
T0*'
_output_shapes
:?
?
-ssd_300_vgg/block11_box/conv_loc/weights/readIdentity(ssd_300_vgg/block11_box/conv_loc/weights*
T0*'
_output_shapes
:?*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights
?
Hssd_300_vgg/block11_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *o:
?
Issd_300_vgg/block11_box/conv_loc/kernel/Regularizer/l2_regularizer/L2LossL2Loss-ssd_300_vgg/block11_box/conv_loc/weights/read*
_output_shapes
: *
T0
?
Bssd_300_vgg/block11_box/conv_loc/kernel/Regularizer/l2_regularizerMulHssd_300_vgg/block11_box/conv_loc/kernel/Regularizer/l2_regularizer/scaleIssd_300_vgg/block11_box/conv_loc/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
9ssd_300_vgg/block11_box/conv_loc/biases/Initializer/zerosConst*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_loc/biases*
_output_shapes
:*
valueB*    *
dtype0
?
'ssd_300_vgg/block11_box/conv_loc/biases
VariableV2*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_loc/biases*
_output_shapes
:*
shape:*
shared_name *
dtype0*
	container 
?
.ssd_300_vgg/block11_box/conv_loc/biases/AssignAssign'ssd_300_vgg/block11_box/conv_loc/biases9ssd_300_vgg/block11_box/conv_loc/biases/Initializer/zeros*
T0*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_loc/biases*
validate_shape(*
_output_shapes
:*
use_locking(
?
,ssd_300_vgg/block11_box/conv_loc/biases/readIdentity'ssd_300_vgg/block11_box/conv_loc/biases*
_output_shapes
:*
T0*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_loc/biases

.ssd_300_vgg/block11_box/conv_loc/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
'ssd_300_vgg/block11_box/conv_loc/Conv2DConv2Dssd_300_vgg/gag-ssd_300_vgg/block11_box/conv_loc/weights/read*&
_output_shapes
:*
paddingSAME*
strides
*
use_cudnn_on_gpu(*
	dilations
*
data_formatNHWC*
T0
?
(ssd_300_vgg/block11_box/conv_loc/BiasAddBiasAdd'ssd_300_vgg/block11_box/conv_loc/Conv2D,ssd_300_vgg/block11_box/conv_loc/biases/read*
T0*
data_formatNHWC*&
_output_shapes
:
?
%ssd_300_vgg/block11_box/Reshape/shapeConst*
_output_shapes
:*)
value B"               *
dtype0
?
ssd_300_vgg/block11_box/ReshapeReshape(ssd_300_vgg/block11_box/conv_loc/BiasAdd%ssd_300_vgg/block11_box/Reshape/shape*
T0**
_output_shapes
:*
Tshape0
?
Issd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*%
valueB"      ?   T   *
dtype0
?
Gssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/minConst*
_output_shapes
: *;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*
valueB
 *9?e?*
dtype0
?
Gssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/maxConst*
valueB
 *9?e=*
dtype0*
_output_shapes
: *;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights
?
Qssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/RandomUniformRandomUniformIssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/shape*

seed *;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*
dtype0*
seed2 *
T0*'
_output_shapes
:?T
?
Gssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/subSubGssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/maxGssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/min*
_output_shapes
: *;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*
T0
?
Gssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/mulMulQssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/RandomUniformGssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/sub*
T0*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*'
_output_shapes
:?T
?
Cssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniformAddGssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/mulGssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform/min*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*
T0*'
_output_shapes
:?T
?
(ssd_300_vgg/block11_box/conv_cls/weights
VariableV2*
shape:?T*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*
	container *
shared_name *
dtype0*'
_output_shapes
:?T
?
/ssd_300_vgg/block11_box/conv_cls/weights/AssignAssign(ssd_300_vgg/block11_box/conv_cls/weightsCssd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform*
validate_shape(*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*
T0*
use_locking(*'
_output_shapes
:?T
?
-ssd_300_vgg/block11_box/conv_cls/weights/readIdentity(ssd_300_vgg/block11_box/conv_cls/weights*
T0*'
_output_shapes
:?T*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights
?
Hssd_300_vgg/block11_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
?
Issd_300_vgg/block11_box/conv_cls/kernel/Regularizer/l2_regularizer/L2LossL2Loss-ssd_300_vgg/block11_box/conv_cls/weights/read*
_output_shapes
: *
T0
?
Bssd_300_vgg/block11_box/conv_cls/kernel/Regularizer/l2_regularizerMulHssd_300_vgg/block11_box/conv_cls/kernel/Regularizer/l2_regularizer/scaleIssd_300_vgg/block11_box/conv_cls/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
?
9ssd_300_vgg/block11_box/conv_cls/biases/Initializer/zerosConst*
valueBT*    *:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_cls/biases*
dtype0*
_output_shapes
:T
?
'ssd_300_vgg/block11_box/conv_cls/biases
VariableV2*
dtype0*
	container *
shape:T*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_cls/biases*
_output_shapes
:T*
shared_name 
?
.ssd_300_vgg/block11_box/conv_cls/biases/AssignAssign'ssd_300_vgg/block11_box/conv_cls/biases9ssd_300_vgg/block11_box/conv_cls/biases/Initializer/zeros*
_output_shapes
:T*
validate_shape(*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_cls/biases*
T0
?
,ssd_300_vgg/block11_box/conv_cls/biases/readIdentity'ssd_300_vgg/block11_box/conv_cls/biases*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_cls/biases*
T0*
_output_shapes
:T

.ssd_300_vgg/block11_box/conv_cls/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
'ssd_300_vgg/block11_box/conv_cls/Conv2DConv2Dssd_300_vgg/gag-ssd_300_vgg/block11_box/conv_cls/weights/read*
paddingSAME*
	dilations
*
data_formatNHWC*
strides
*&
_output_shapes
:T*
use_cudnn_on_gpu(*
T0
?
(ssd_300_vgg/block11_box/conv_cls/BiasAddBiasAdd'ssd_300_vgg/block11_box/conv_cls/Conv2D,ssd_300_vgg/block11_box/conv_cls/biases/read*&
_output_shapes
:T*
data_formatNHWC*
T0
?
'ssd_300_vgg/block11_box/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*)
value B"               
?
!ssd_300_vgg/block11_box/Reshape_1Reshape(ssd_300_vgg/block11_box/conv_cls/BiasAdd'ssd_300_vgg/block11_box/Reshape_1/shape*
T0**
_output_shapes
:*
Tshape0
t
#ssd_300_vgg/softmax_5/Reshape/shapeConst*
dtype0*
valueB"????   *
_output_shapes
:
?
ssd_300_vgg/softmax_5/ReshapeReshape!ssd_300_vgg/block11_box/Reshape_1#ssd_300_vgg/softmax_5/Reshape/shape*
_output_shapes

:*
Tshape0*
T0
p
ssd_300_vgg/softmax_5/SoftmaxSoftmaxssd_300_vgg/softmax_5/Reshape*
T0*
_output_shapes

:
x
ssd_300_vgg/softmax_5/ShapeConst*)
value B"               *
dtype0*
_output_shapes
:
?
ssd_300_vgg/softmax_5/Reshape_1Reshapessd_300_vgg/softmax_5/Softmaxssd_300_vgg/softmax_5/Shape*
Tshape0*
T0**
_output_shapes
:
?:
initNoOp&^ssd_300_vgg/batch_norm_00/beta/Assign'^ssd_300_vgg/batch_norm_00/gamma/Assign-^ssd_300_vgg/batch_norm_00/moving_mean/Assign1^ssd_300_vgg/batch_norm_00/moving_variance/Assign/^ssd_300_vgg/block10_box/conv_cls/biases/Assign0^ssd_300_vgg/block10_box/conv_cls/weights/Assign/^ssd_300_vgg/block10_box/conv_loc/biases/Assign0^ssd_300_vgg/block10_box/conv_loc/weights/Assign/^ssd_300_vgg/block11_box/conv_cls/biases/Assign0^ssd_300_vgg/block11_box/conv_cls/weights/Assign/^ssd_300_vgg/block11_box/conv_loc/biases/Assign0^ssd_300_vgg/block11_box/conv_loc/weights/Assign4^ssd_300_vgg/block4_box/L2Normalization/gamma/Assign.^ssd_300_vgg/block4_box/conv_cls/biases/Assign/^ssd_300_vgg/block4_box/conv_cls/weights/Assign.^ssd_300_vgg/block4_box/conv_loc/biases/Assign/^ssd_300_vgg/block4_box/conv_loc/weights/Assign.^ssd_300_vgg/block7_box/conv_cls/biases/Assign/^ssd_300_vgg/block7_box/conv_cls/weights/Assign.^ssd_300_vgg/block7_box/conv_loc/biases/Assign/^ssd_300_vgg/block7_box/conv_loc/weights/Assign.^ssd_300_vgg/block8_box/conv_cls/biases/Assign/^ssd_300_vgg/block8_box/conv_cls/weights/Assign.^ssd_300_vgg/block8_box/conv_loc/biases/Assign/^ssd_300_vgg/block8_box/conv_loc/weights/Assign.^ssd_300_vgg/block9_box/conv_cls/biases/Assign/^ssd_300_vgg/block9_box/conv_cls/weights/Assign.^ssd_300_vgg/block9_box/conv_loc/biases/Assign/^ssd_300_vgg/block9_box/conv_loc/weights/Assign#^ssd_300_vgg/conv10_1/biases/Assign$^ssd_300_vgg/conv10_1/weights/Assign#^ssd_300_vgg/conv10_2/biases/Assign$^ssd_300_vgg/conv10_2/weights/Assign"^ssd_300_vgg/conv8_1/biases/Assign#^ssd_300_vgg/conv8_1/weights/Assign"^ssd_300_vgg/conv8_2/biases/Assign#^ssd_300_vgg/conv8_2/weights/Assign"^ssd_300_vgg/conv9_1/biases/Assign#^ssd_300_vgg/conv9_1/weights/Assign"^ssd_300_vgg/conv9_2/biases/Assign#^ssd_300_vgg/conv9_2/weights/Assign$^ssd_300_vgg/conv_init/biases/Assign%^ssd_300_vgg/conv_init/weights/Assign&^ssd_300_vgg/conv_init_1/biases/Assign'^ssd_300_vgg/conv_init_1/weights/Assign1^ssd_300_vgg/resblock0_0/batch_norm_0/beta/Assign2^ssd_300_vgg/resblock0_0/batch_norm_0/gamma/Assign8^ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/Assign<^ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/Assign1^ssd_300_vgg/resblock0_0/batch_norm_1/beta/Assign2^ssd_300_vgg/resblock0_0/batch_norm_1/gamma/Assign8^ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/Assign<^ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/Assign-^ssd_300_vgg/resblock0_0/conv_0/biases/Assign.^ssd_300_vgg/resblock0_0/conv_0/weights/Assign-^ssd_300_vgg/resblock0_0/conv_1/biases/Assign.^ssd_300_vgg/resblock0_0/conv_1/weights/Assign1^ssd_300_vgg/resblock0_1/batch_norm_0/beta/Assign2^ssd_300_vgg/resblock0_1/batch_norm_0/gamma/Assign8^ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/Assign<^ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/Assign1^ssd_300_vgg/resblock0_1/batch_norm_1/beta/Assign2^ssd_300_vgg/resblock0_1/batch_norm_1/gamma/Assign8^ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/Assign<^ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/Assign-^ssd_300_vgg/resblock0_1/conv_0/biases/Assign.^ssd_300_vgg/resblock0_1/conv_0/weights/Assign-^ssd_300_vgg/resblock0_1/conv_1/biases/Assign.^ssd_300_vgg/resblock0_1/conv_1/weights/Assign1^ssd_300_vgg/resblock1_0/batch_norm_0/beta/Assign2^ssd_300_vgg/resblock1_0/batch_norm_0/gamma/Assign8^ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/Assign<^ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/Assign1^ssd_300_vgg/resblock1_0/batch_norm_1/beta/Assign2^ssd_300_vgg/resblock1_0/batch_norm_1/gamma/Assign8^ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/Assign<^ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/Assign-^ssd_300_vgg/resblock1_0/conv_0/biases/Assign.^ssd_300_vgg/resblock1_0/conv_0/weights/Assign-^ssd_300_vgg/resblock1_0/conv_1/biases/Assign.^ssd_300_vgg/resblock1_0/conv_1/weights/Assign0^ssd_300_vgg/resblock1_0/conv_init/biases/Assign1^ssd_300_vgg/resblock1_0/conv_init/weights/Assign1^ssd_300_vgg/resblock1_1/batch_norm_0/beta/Assign2^ssd_300_vgg/resblock1_1/batch_norm_0/gamma/Assign8^ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/Assign<^ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/Assign1^ssd_300_vgg/resblock1_1/batch_norm_1/beta/Assign2^ssd_300_vgg/resblock1_1/batch_norm_1/gamma/Assign8^ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/Assign<^ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/Assign-^ssd_300_vgg/resblock1_1/conv_0/biases/Assign.^ssd_300_vgg/resblock1_1/conv_0/weights/Assign-^ssd_300_vgg/resblock1_1/conv_1/biases/Assign.^ssd_300_vgg/resblock1_1/conv_1/weights/Assign1^ssd_300_vgg/resblock2_0/batch_norm_0/beta/Assign2^ssd_300_vgg/resblock2_0/batch_norm_0/gamma/Assign8^ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/Assign<^ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/Assign1^ssd_300_vgg/resblock2_0/batch_norm_1/beta/Assign2^ssd_300_vgg/resblock2_0/batch_norm_1/gamma/Assign8^ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/Assign<^ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/Assign-^ssd_300_vgg/resblock2_0/conv_0/biases/Assign.^ssd_300_vgg/resblock2_0/conv_0/weights/Assign-^ssd_300_vgg/resblock2_0/conv_1/biases/Assign.^ssd_300_vgg/resblock2_0/conv_1/weights/Assign0^ssd_300_vgg/resblock2_0/conv_init/biases/Assign1^ssd_300_vgg/resblock2_0/conv_init/weights/Assign1^ssd_300_vgg/resblock2_1/batch_norm_0/beta/Assign2^ssd_300_vgg/resblock2_1/batch_norm_0/gamma/Assign8^ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/Assign<^ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/Assign1^ssd_300_vgg/resblock2_1/batch_norm_1/beta/Assign2^ssd_300_vgg/resblock2_1/batch_norm_1/gamma/Assign8^ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/Assign<^ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/Assign-^ssd_300_vgg/resblock2_1/conv_0/biases/Assign.^ssd_300_vgg/resblock2_1/conv_0/weights/Assign-^ssd_300_vgg/resblock2_1/conv_1/biases/Assign.^ssd_300_vgg/resblock2_1/conv_1/weights/Assign2^ssd_300_vgg/resblock_3_0/batch_norm_0/beta/Assign3^ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/Assign9^ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/Assign=^ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/Assign2^ssd_300_vgg/resblock_3_0/batch_norm_1/beta/Assign3^ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/Assign9^ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/Assign=^ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/Assign.^ssd_300_vgg/resblock_3_0/conv_0/biases/Assign/^ssd_300_vgg/resblock_3_0/conv_0/weights/Assign.^ssd_300_vgg/resblock_3_0/conv_1/biases/Assign/^ssd_300_vgg/resblock_3_0/conv_1/weights/Assign1^ssd_300_vgg/resblock_3_0/conv_init/biases/Assign2^ssd_300_vgg/resblock_3_0/conv_init/weights/Assign2^ssd_300_vgg/resblock_3_1/batch_norm_0/beta/Assign3^ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/Assign9^ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/Assign=^ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/Assign2^ssd_300_vgg/resblock_3_1/batch_norm_1/beta/Assign3^ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/Assign9^ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/Assign=^ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/Assign.^ssd_300_vgg/resblock_3_1/conv_0/biases/Assign/^ssd_300_vgg/resblock_3_1/conv_0/weights/Assign.^ssd_300_vgg/resblock_3_1/conv_1/biases/Assign/^ssd_300_vgg/resblock_3_1/conv_1/weights/Assign
Y
save/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
?1
save/SaveV2/tensor_namesConst*
dtype0*?0
value?0B?0?Bssd_300_vgg/batch_norm_00/betaBssd_300_vgg/batch_norm_00/gammaB%ssd_300_vgg/batch_norm_00/moving_meanB)ssd_300_vgg/batch_norm_00/moving_varianceB'ssd_300_vgg/block10_box/conv_cls/biasesB(ssd_300_vgg/block10_box/conv_cls/weightsB'ssd_300_vgg/block10_box/conv_loc/biasesB(ssd_300_vgg/block10_box/conv_loc/weightsB'ssd_300_vgg/block11_box/conv_cls/biasesB(ssd_300_vgg/block11_box/conv_cls/weightsB'ssd_300_vgg/block11_box/conv_loc/biasesB(ssd_300_vgg/block11_box/conv_loc/weightsB,ssd_300_vgg/block4_box/L2Normalization/gammaB&ssd_300_vgg/block4_box/conv_cls/biasesB'ssd_300_vgg/block4_box/conv_cls/weightsB&ssd_300_vgg/block4_box/conv_loc/biasesB'ssd_300_vgg/block4_box/conv_loc/weightsB&ssd_300_vgg/block7_box/conv_cls/biasesB'ssd_300_vgg/block7_box/conv_cls/weightsB&ssd_300_vgg/block7_box/conv_loc/biasesB'ssd_300_vgg/block7_box/conv_loc/weightsB&ssd_300_vgg/block8_box/conv_cls/biasesB'ssd_300_vgg/block8_box/conv_cls/weightsB&ssd_300_vgg/block8_box/conv_loc/biasesB'ssd_300_vgg/block8_box/conv_loc/weightsB&ssd_300_vgg/block9_box/conv_cls/biasesB'ssd_300_vgg/block9_box/conv_cls/weightsB&ssd_300_vgg/block9_box/conv_loc/biasesB'ssd_300_vgg/block9_box/conv_loc/weightsBssd_300_vgg/conv10_1/biasesBssd_300_vgg/conv10_1/weightsBssd_300_vgg/conv10_2/biasesBssd_300_vgg/conv10_2/weightsBssd_300_vgg/conv8_1/biasesBssd_300_vgg/conv8_1/weightsBssd_300_vgg/conv8_2/biasesBssd_300_vgg/conv8_2/weightsBssd_300_vgg/conv9_1/biasesBssd_300_vgg/conv9_1/weightsBssd_300_vgg/conv9_2/biasesBssd_300_vgg/conv9_2/weightsBssd_300_vgg/conv_init/biasesBssd_300_vgg/conv_init/weightsBssd_300_vgg/conv_init_1/biasesBssd_300_vgg/conv_init_1/weightsB)ssd_300_vgg/resblock0_0/batch_norm_0/betaB*ssd_300_vgg/resblock0_0/batch_norm_0/gammaB0ssd_300_vgg/resblock0_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock0_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock0_0/batch_norm_1/betaB*ssd_300_vgg/resblock0_0/batch_norm_1/gammaB0ssd_300_vgg/resblock0_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock0_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock0_0/conv_0/biasesB&ssd_300_vgg/resblock0_0/conv_0/weightsB%ssd_300_vgg/resblock0_0/conv_1/biasesB&ssd_300_vgg/resblock0_0/conv_1/weightsB)ssd_300_vgg/resblock0_1/batch_norm_0/betaB*ssd_300_vgg/resblock0_1/batch_norm_0/gammaB0ssd_300_vgg/resblock0_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock0_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock0_1/batch_norm_1/betaB*ssd_300_vgg/resblock0_1/batch_norm_1/gammaB0ssd_300_vgg/resblock0_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock0_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock0_1/conv_0/biasesB&ssd_300_vgg/resblock0_1/conv_0/weightsB%ssd_300_vgg/resblock0_1/conv_1/biasesB&ssd_300_vgg/resblock0_1/conv_1/weightsB)ssd_300_vgg/resblock1_0/batch_norm_0/betaB*ssd_300_vgg/resblock1_0/batch_norm_0/gammaB0ssd_300_vgg/resblock1_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock1_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock1_0/batch_norm_1/betaB*ssd_300_vgg/resblock1_0/batch_norm_1/gammaB0ssd_300_vgg/resblock1_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock1_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock1_0/conv_0/biasesB&ssd_300_vgg/resblock1_0/conv_0/weightsB%ssd_300_vgg/resblock1_0/conv_1/biasesB&ssd_300_vgg/resblock1_0/conv_1/weightsB(ssd_300_vgg/resblock1_0/conv_init/biasesB)ssd_300_vgg/resblock1_0/conv_init/weightsB)ssd_300_vgg/resblock1_1/batch_norm_0/betaB*ssd_300_vgg/resblock1_1/batch_norm_0/gammaB0ssd_300_vgg/resblock1_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock1_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock1_1/batch_norm_1/betaB*ssd_300_vgg/resblock1_1/batch_norm_1/gammaB0ssd_300_vgg/resblock1_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock1_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock1_1/conv_0/biasesB&ssd_300_vgg/resblock1_1/conv_0/weightsB%ssd_300_vgg/resblock1_1/conv_1/biasesB&ssd_300_vgg/resblock1_1/conv_1/weightsB)ssd_300_vgg/resblock2_0/batch_norm_0/betaB*ssd_300_vgg/resblock2_0/batch_norm_0/gammaB0ssd_300_vgg/resblock2_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock2_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock2_0/batch_norm_1/betaB*ssd_300_vgg/resblock2_0/batch_norm_1/gammaB0ssd_300_vgg/resblock2_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock2_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock2_0/conv_0/biasesB&ssd_300_vgg/resblock2_0/conv_0/weightsB%ssd_300_vgg/resblock2_0/conv_1/biasesB&ssd_300_vgg/resblock2_0/conv_1/weightsB(ssd_300_vgg/resblock2_0/conv_init/biasesB)ssd_300_vgg/resblock2_0/conv_init/weightsB)ssd_300_vgg/resblock2_1/batch_norm_0/betaB*ssd_300_vgg/resblock2_1/batch_norm_0/gammaB0ssd_300_vgg/resblock2_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock2_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock2_1/batch_norm_1/betaB*ssd_300_vgg/resblock2_1/batch_norm_1/gammaB0ssd_300_vgg/resblock2_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock2_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock2_1/conv_0/biasesB&ssd_300_vgg/resblock2_1/conv_0/weightsB%ssd_300_vgg/resblock2_1/conv_1/biasesB&ssd_300_vgg/resblock2_1/conv_1/weightsB*ssd_300_vgg/resblock_3_0/batch_norm_0/betaB+ssd_300_vgg/resblock_3_0/batch_norm_0/gammaB1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_meanB5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_varianceB*ssd_300_vgg/resblock_3_0/batch_norm_1/betaB+ssd_300_vgg/resblock_3_0/batch_norm_1/gammaB1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_meanB5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_varianceB&ssd_300_vgg/resblock_3_0/conv_0/biasesB'ssd_300_vgg/resblock_3_0/conv_0/weightsB&ssd_300_vgg/resblock_3_0/conv_1/biasesB'ssd_300_vgg/resblock_3_0/conv_1/weightsB)ssd_300_vgg/resblock_3_0/conv_init/biasesB*ssd_300_vgg/resblock_3_0/conv_init/weightsB*ssd_300_vgg/resblock_3_1/batch_norm_0/betaB+ssd_300_vgg/resblock_3_1/batch_norm_0/gammaB1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_meanB5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_varianceB*ssd_300_vgg/resblock_3_1/batch_norm_1/betaB+ssd_300_vgg/resblock_3_1/batch_norm_1/gammaB1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_meanB5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_varianceB&ssd_300_vgg/resblock_3_1/conv_0/biasesB'ssd_300_vgg/resblock_3_1/conv_0/weightsB&ssd_300_vgg/resblock_3_1/conv_1/biasesB'ssd_300_vgg/resblock_3_1/conv_1/weights*
_output_shapes	
:?
?
save/SaveV2/shape_and_slicesConst*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?2
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesssd_300_vgg/batch_norm_00/betassd_300_vgg/batch_norm_00/gamma%ssd_300_vgg/batch_norm_00/moving_mean)ssd_300_vgg/batch_norm_00/moving_variance'ssd_300_vgg/block10_box/conv_cls/biases(ssd_300_vgg/block10_box/conv_cls/weights'ssd_300_vgg/block10_box/conv_loc/biases(ssd_300_vgg/block10_box/conv_loc/weights'ssd_300_vgg/block11_box/conv_cls/biases(ssd_300_vgg/block11_box/conv_cls/weights'ssd_300_vgg/block11_box/conv_loc/biases(ssd_300_vgg/block11_box/conv_loc/weights,ssd_300_vgg/block4_box/L2Normalization/gamma&ssd_300_vgg/block4_box/conv_cls/biases'ssd_300_vgg/block4_box/conv_cls/weights&ssd_300_vgg/block4_box/conv_loc/biases'ssd_300_vgg/block4_box/conv_loc/weights&ssd_300_vgg/block7_box/conv_cls/biases'ssd_300_vgg/block7_box/conv_cls/weights&ssd_300_vgg/block7_box/conv_loc/biases'ssd_300_vgg/block7_box/conv_loc/weights&ssd_300_vgg/block8_box/conv_cls/biases'ssd_300_vgg/block8_box/conv_cls/weights&ssd_300_vgg/block8_box/conv_loc/biases'ssd_300_vgg/block8_box/conv_loc/weights&ssd_300_vgg/block9_box/conv_cls/biases'ssd_300_vgg/block9_box/conv_cls/weights&ssd_300_vgg/block9_box/conv_loc/biases'ssd_300_vgg/block9_box/conv_loc/weightsssd_300_vgg/conv10_1/biasesssd_300_vgg/conv10_1/weightsssd_300_vgg/conv10_2/biasesssd_300_vgg/conv10_2/weightsssd_300_vgg/conv8_1/biasesssd_300_vgg/conv8_1/weightsssd_300_vgg/conv8_2/biasesssd_300_vgg/conv8_2/weightsssd_300_vgg/conv9_1/biasesssd_300_vgg/conv9_1/weightsssd_300_vgg/conv9_2/biasesssd_300_vgg/conv9_2/weightsssd_300_vgg/conv_init/biasesssd_300_vgg/conv_init/weightsssd_300_vgg/conv_init_1/biasesssd_300_vgg/conv_init_1/weights)ssd_300_vgg/resblock0_0/batch_norm_0/beta*ssd_300_vgg/resblock0_0/batch_norm_0/gamma0ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean4ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance)ssd_300_vgg/resblock0_0/batch_norm_1/beta*ssd_300_vgg/resblock0_0/batch_norm_1/gamma0ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean4ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance%ssd_300_vgg/resblock0_0/conv_0/biases&ssd_300_vgg/resblock0_0/conv_0/weights%ssd_300_vgg/resblock0_0/conv_1/biases&ssd_300_vgg/resblock0_0/conv_1/weights)ssd_300_vgg/resblock0_1/batch_norm_0/beta*ssd_300_vgg/resblock0_1/batch_norm_0/gamma0ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean4ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance)ssd_300_vgg/resblock0_1/batch_norm_1/beta*ssd_300_vgg/resblock0_1/batch_norm_1/gamma0ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean4ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance%ssd_300_vgg/resblock0_1/conv_0/biases&ssd_300_vgg/resblock0_1/conv_0/weights%ssd_300_vgg/resblock0_1/conv_1/biases&ssd_300_vgg/resblock0_1/conv_1/weights)ssd_300_vgg/resblock1_0/batch_norm_0/beta*ssd_300_vgg/resblock1_0/batch_norm_0/gamma0ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean4ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance)ssd_300_vgg/resblock1_0/batch_norm_1/beta*ssd_300_vgg/resblock1_0/batch_norm_1/gamma0ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean4ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance%ssd_300_vgg/resblock1_0/conv_0/biases&ssd_300_vgg/resblock1_0/conv_0/weights%ssd_300_vgg/resblock1_0/conv_1/biases&ssd_300_vgg/resblock1_0/conv_1/weights(ssd_300_vgg/resblock1_0/conv_init/biases)ssd_300_vgg/resblock1_0/conv_init/weights)ssd_300_vgg/resblock1_1/batch_norm_0/beta*ssd_300_vgg/resblock1_1/batch_norm_0/gamma0ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean4ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance)ssd_300_vgg/resblock1_1/batch_norm_1/beta*ssd_300_vgg/resblock1_1/batch_norm_1/gamma0ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean4ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance%ssd_300_vgg/resblock1_1/conv_0/biases&ssd_300_vgg/resblock1_1/conv_0/weights%ssd_300_vgg/resblock1_1/conv_1/biases&ssd_300_vgg/resblock1_1/conv_1/weights)ssd_300_vgg/resblock2_0/batch_norm_0/beta*ssd_300_vgg/resblock2_0/batch_norm_0/gamma0ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean4ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance)ssd_300_vgg/resblock2_0/batch_norm_1/beta*ssd_300_vgg/resblock2_0/batch_norm_1/gamma0ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean4ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance%ssd_300_vgg/resblock2_0/conv_0/biases&ssd_300_vgg/resblock2_0/conv_0/weights%ssd_300_vgg/resblock2_0/conv_1/biases&ssd_300_vgg/resblock2_0/conv_1/weights(ssd_300_vgg/resblock2_0/conv_init/biases)ssd_300_vgg/resblock2_0/conv_init/weights)ssd_300_vgg/resblock2_1/batch_norm_0/beta*ssd_300_vgg/resblock2_1/batch_norm_0/gamma0ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean4ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance)ssd_300_vgg/resblock2_1/batch_norm_1/beta*ssd_300_vgg/resblock2_1/batch_norm_1/gamma0ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean4ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance%ssd_300_vgg/resblock2_1/conv_0/biases&ssd_300_vgg/resblock2_1/conv_0/weights%ssd_300_vgg/resblock2_1/conv_1/biases&ssd_300_vgg/resblock2_1/conv_1/weights*ssd_300_vgg/resblock_3_0/batch_norm_0/beta+ssd_300_vgg/resblock_3_0/batch_norm_0/gamma1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance*ssd_300_vgg/resblock_3_0/batch_norm_1/beta+ssd_300_vgg/resblock_3_0/batch_norm_1/gamma1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance&ssd_300_vgg/resblock_3_0/conv_0/biases'ssd_300_vgg/resblock_3_0/conv_0/weights&ssd_300_vgg/resblock_3_0/conv_1/biases'ssd_300_vgg/resblock_3_0/conv_1/weights)ssd_300_vgg/resblock_3_0/conv_init/biases*ssd_300_vgg/resblock_3_0/conv_init/weights*ssd_300_vgg/resblock_3_1/batch_norm_0/beta+ssd_300_vgg/resblock_3_1/batch_norm_0/gamma1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance*ssd_300_vgg/resblock_3_1/batch_norm_1/beta+ssd_300_vgg/resblock_3_1/batch_norm_1/gamma1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance&ssd_300_vgg/resblock_3_1/conv_0/biases'ssd_300_vgg/resblock_3_1/conv_0/weights&ssd_300_vgg/resblock_3_1/conv_1/biases'ssd_300_vgg/resblock_3_1/conv_1/weights*?
dtypes?
?2?
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
?1
save/RestoreV2/tensor_namesConst*
_output_shapes	
:?*?0
value?0B?0?Bssd_300_vgg/batch_norm_00/betaBssd_300_vgg/batch_norm_00/gammaB%ssd_300_vgg/batch_norm_00/moving_meanB)ssd_300_vgg/batch_norm_00/moving_varianceB'ssd_300_vgg/block10_box/conv_cls/biasesB(ssd_300_vgg/block10_box/conv_cls/weightsB'ssd_300_vgg/block10_box/conv_loc/biasesB(ssd_300_vgg/block10_box/conv_loc/weightsB'ssd_300_vgg/block11_box/conv_cls/biasesB(ssd_300_vgg/block11_box/conv_cls/weightsB'ssd_300_vgg/block11_box/conv_loc/biasesB(ssd_300_vgg/block11_box/conv_loc/weightsB,ssd_300_vgg/block4_box/L2Normalization/gammaB&ssd_300_vgg/block4_box/conv_cls/biasesB'ssd_300_vgg/block4_box/conv_cls/weightsB&ssd_300_vgg/block4_box/conv_loc/biasesB'ssd_300_vgg/block4_box/conv_loc/weightsB&ssd_300_vgg/block7_box/conv_cls/biasesB'ssd_300_vgg/block7_box/conv_cls/weightsB&ssd_300_vgg/block7_box/conv_loc/biasesB'ssd_300_vgg/block7_box/conv_loc/weightsB&ssd_300_vgg/block8_box/conv_cls/biasesB'ssd_300_vgg/block8_box/conv_cls/weightsB&ssd_300_vgg/block8_box/conv_loc/biasesB'ssd_300_vgg/block8_box/conv_loc/weightsB&ssd_300_vgg/block9_box/conv_cls/biasesB'ssd_300_vgg/block9_box/conv_cls/weightsB&ssd_300_vgg/block9_box/conv_loc/biasesB'ssd_300_vgg/block9_box/conv_loc/weightsBssd_300_vgg/conv10_1/biasesBssd_300_vgg/conv10_1/weightsBssd_300_vgg/conv10_2/biasesBssd_300_vgg/conv10_2/weightsBssd_300_vgg/conv8_1/biasesBssd_300_vgg/conv8_1/weightsBssd_300_vgg/conv8_2/biasesBssd_300_vgg/conv8_2/weightsBssd_300_vgg/conv9_1/biasesBssd_300_vgg/conv9_1/weightsBssd_300_vgg/conv9_2/biasesBssd_300_vgg/conv9_2/weightsBssd_300_vgg/conv_init/biasesBssd_300_vgg/conv_init/weightsBssd_300_vgg/conv_init_1/biasesBssd_300_vgg/conv_init_1/weightsB)ssd_300_vgg/resblock0_0/batch_norm_0/betaB*ssd_300_vgg/resblock0_0/batch_norm_0/gammaB0ssd_300_vgg/resblock0_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock0_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock0_0/batch_norm_1/betaB*ssd_300_vgg/resblock0_0/batch_norm_1/gammaB0ssd_300_vgg/resblock0_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock0_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock0_0/conv_0/biasesB&ssd_300_vgg/resblock0_0/conv_0/weightsB%ssd_300_vgg/resblock0_0/conv_1/biasesB&ssd_300_vgg/resblock0_0/conv_1/weightsB)ssd_300_vgg/resblock0_1/batch_norm_0/betaB*ssd_300_vgg/resblock0_1/batch_norm_0/gammaB0ssd_300_vgg/resblock0_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock0_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock0_1/batch_norm_1/betaB*ssd_300_vgg/resblock0_1/batch_norm_1/gammaB0ssd_300_vgg/resblock0_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock0_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock0_1/conv_0/biasesB&ssd_300_vgg/resblock0_1/conv_0/weightsB%ssd_300_vgg/resblock0_1/conv_1/biasesB&ssd_300_vgg/resblock0_1/conv_1/weightsB)ssd_300_vgg/resblock1_0/batch_norm_0/betaB*ssd_300_vgg/resblock1_0/batch_norm_0/gammaB0ssd_300_vgg/resblock1_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock1_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock1_0/batch_norm_1/betaB*ssd_300_vgg/resblock1_0/batch_norm_1/gammaB0ssd_300_vgg/resblock1_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock1_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock1_0/conv_0/biasesB&ssd_300_vgg/resblock1_0/conv_0/weightsB%ssd_300_vgg/resblock1_0/conv_1/biasesB&ssd_300_vgg/resblock1_0/conv_1/weightsB(ssd_300_vgg/resblock1_0/conv_init/biasesB)ssd_300_vgg/resblock1_0/conv_init/weightsB)ssd_300_vgg/resblock1_1/batch_norm_0/betaB*ssd_300_vgg/resblock1_1/batch_norm_0/gammaB0ssd_300_vgg/resblock1_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock1_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock1_1/batch_norm_1/betaB*ssd_300_vgg/resblock1_1/batch_norm_1/gammaB0ssd_300_vgg/resblock1_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock1_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock1_1/conv_0/biasesB&ssd_300_vgg/resblock1_1/conv_0/weightsB%ssd_300_vgg/resblock1_1/conv_1/biasesB&ssd_300_vgg/resblock1_1/conv_1/weightsB)ssd_300_vgg/resblock2_0/batch_norm_0/betaB*ssd_300_vgg/resblock2_0/batch_norm_0/gammaB0ssd_300_vgg/resblock2_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock2_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock2_0/batch_norm_1/betaB*ssd_300_vgg/resblock2_0/batch_norm_1/gammaB0ssd_300_vgg/resblock2_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock2_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock2_0/conv_0/biasesB&ssd_300_vgg/resblock2_0/conv_0/weightsB%ssd_300_vgg/resblock2_0/conv_1/biasesB&ssd_300_vgg/resblock2_0/conv_1/weightsB(ssd_300_vgg/resblock2_0/conv_init/biasesB)ssd_300_vgg/resblock2_0/conv_init/weightsB)ssd_300_vgg/resblock2_1/batch_norm_0/betaB*ssd_300_vgg/resblock2_1/batch_norm_0/gammaB0ssd_300_vgg/resblock2_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock2_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock2_1/batch_norm_1/betaB*ssd_300_vgg/resblock2_1/batch_norm_1/gammaB0ssd_300_vgg/resblock2_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock2_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock2_1/conv_0/biasesB&ssd_300_vgg/resblock2_1/conv_0/weightsB%ssd_300_vgg/resblock2_1/conv_1/biasesB&ssd_300_vgg/resblock2_1/conv_1/weightsB*ssd_300_vgg/resblock_3_0/batch_norm_0/betaB+ssd_300_vgg/resblock_3_0/batch_norm_0/gammaB1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_meanB5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_varianceB*ssd_300_vgg/resblock_3_0/batch_norm_1/betaB+ssd_300_vgg/resblock_3_0/batch_norm_1/gammaB1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_meanB5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_varianceB&ssd_300_vgg/resblock_3_0/conv_0/biasesB'ssd_300_vgg/resblock_3_0/conv_0/weightsB&ssd_300_vgg/resblock_3_0/conv_1/biasesB'ssd_300_vgg/resblock_3_0/conv_1/weightsB)ssd_300_vgg/resblock_3_0/conv_init/biasesB*ssd_300_vgg/resblock_3_0/conv_init/weightsB*ssd_300_vgg/resblock_3_1/batch_norm_0/betaB+ssd_300_vgg/resblock_3_1/batch_norm_0/gammaB1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_meanB5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_varianceB*ssd_300_vgg/resblock_3_1/batch_norm_1/betaB+ssd_300_vgg/resblock_3_1/batch_norm_1/gammaB1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_meanB5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_varianceB&ssd_300_vgg/resblock_3_1/conv_0/biasesB'ssd_300_vgg/resblock_3_1/conv_0/weightsB&ssd_300_vgg/resblock_3_1/conv_1/biasesB'ssd_300_vgg/resblock_3_1/conv_1/weights*
dtype0
?
save/RestoreV2/shape_and_slicesConst*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:?
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*?
dtypes?
?2?*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save/AssignAssignssd_300_vgg/batch_norm_00/betasave/RestoreV2*
T0*
validate_shape(*
_output_shapes
:@*1
_class'
%#loc:@ssd_300_vgg/batch_norm_00/beta*
use_locking(
?
save/Assign_1Assignssd_300_vgg/batch_norm_00/gammasave/RestoreV2:1*
use_locking(*
T0*
_output_shapes
:@*2
_class(
&$loc:@ssd_300_vgg/batch_norm_00/gamma*
validate_shape(
?
save/Assign_2Assign%ssd_300_vgg/batch_norm_00/moving_meansave/RestoreV2:2*
T0*
use_locking(*
_output_shapes
:@*8
_class.
,*loc:@ssd_300_vgg/batch_norm_00/moving_mean*
validate_shape(
?
save/Assign_3Assign)ssd_300_vgg/batch_norm_00/moving_variancesave/RestoreV2:3*<
_class2
0.loc:@ssd_300_vgg/batch_norm_00/moving_variance*
use_locking(*
validate_shape(*
_output_shapes
:@*
T0
?
save/Assign_4Assign'ssd_300_vgg/block10_box/conv_cls/biasessave/RestoreV2:4*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_cls/biases*
use_locking(*
_output_shapes
:T*
validate_shape(*
T0
?
save/Assign_5Assign(ssd_300_vgg/block10_box/conv_cls/weightssave/RestoreV2:5*
T0*
use_locking(*'
_output_shapes
:?T*
validate_shape(*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights
?
save/Assign_6Assign'ssd_300_vgg/block10_box/conv_loc/biasessave/RestoreV2:6*
_output_shapes
:*
validate_shape(*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_loc/biases*
T0
?
save/Assign_7Assign(ssd_300_vgg/block10_box/conv_loc/weightssave/RestoreV2:7*
use_locking(*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*
T0*'
_output_shapes
:?*
validate_shape(
?
save/Assign_8Assign'ssd_300_vgg/block11_box/conv_cls/biasessave/RestoreV2:8*
validate_shape(*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_cls/biases*
T0*
use_locking(*
_output_shapes
:T
?
save/Assign_9Assign(ssd_300_vgg/block11_box/conv_cls/weightssave/RestoreV2:9*
T0*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*
use_locking(*
validate_shape(*'
_output_shapes
:?T
?
save/Assign_10Assign'ssd_300_vgg/block11_box/conv_loc/biasessave/RestoreV2:10*
_output_shapes
:*
validate_shape(*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_loc/biases*
T0*
use_locking(
?
save/Assign_11Assign(ssd_300_vgg/block11_box/conv_loc/weightssave/RestoreV2:11*
use_locking(*
T0*'
_output_shapes
:?*
validate_shape(*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights
?
save/Assign_12Assign,ssd_300_vgg/block4_box/L2Normalization/gammasave/RestoreV2:12*?
_class5
31loc:@ssd_300_vgg/block4_box/L2Normalization/gamma*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(
?
save/Assign_13Assign&ssd_300_vgg/block4_box/conv_cls/biasessave/RestoreV2:13*
T0*
_output_shapes
:T*9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_cls/biases*
use_locking(*
validate_shape(
?
save/Assign_14Assign'ssd_300_vgg/block4_box/conv_cls/weightssave/RestoreV2:14*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights*'
_output_shapes
:?T*
T0*
use_locking(*
validate_shape(
?
save/Assign_15Assign&ssd_300_vgg/block4_box/conv_loc/biasessave/RestoreV2:15*
_output_shapes
:*9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_loc/biases*
T0*
use_locking(*
validate_shape(
?
save/Assign_16Assign'ssd_300_vgg/block4_box/conv_loc/weightssave/RestoreV2:16*
T0*'
_output_shapes
:?*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
validate_shape(
?
save/Assign_17Assign&ssd_300_vgg/block7_box/conv_cls/biasessave/RestoreV2:17*
T0*9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_cls/biases*
_output_shapes
:~*
use_locking(*
validate_shape(
?
save/Assign_18Assign'ssd_300_vgg/block7_box/conv_cls/weightssave/RestoreV2:18*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights*
validate_shape(*'
_output_shapes
:?~*
T0*
use_locking(
?
save/Assign_19Assign&ssd_300_vgg/block7_box/conv_loc/biasessave/RestoreV2:19*9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_loc/biases*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
?
save/Assign_20Assign'ssd_300_vgg/block7_box/conv_loc/weightssave/RestoreV2:20*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights*
use_locking(*'
_output_shapes
:?*
validate_shape(*
T0
?
save/Assign_21Assign&ssd_300_vgg/block8_box/conv_cls/biasessave/RestoreV2:21*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_cls/biases*
validate_shape(*
T0*
use_locking(*
_output_shapes
:~
?
save/Assign_22Assign'ssd_300_vgg/block8_box/conv_cls/weightssave/RestoreV2:22*
validate_shape(*'
_output_shapes
:?~*
T0*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights
?
save/Assign_23Assign&ssd_300_vgg/block8_box/conv_loc/biasessave/RestoreV2:23*
_output_shapes
:*
T0*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_loc/biases*
validate_shape(
?
save/Assign_24Assign'ssd_300_vgg/block8_box/conv_loc/weightssave/RestoreV2:24*
use_locking(*
T0*
validate_shape(*'
_output_shapes
:?*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights
?
save/Assign_25Assign&ssd_300_vgg/block9_box/conv_cls/biasessave/RestoreV2:25*
validate_shape(*
_output_shapes
:~*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_cls/biases*
T0
?
save/Assign_26Assign'ssd_300_vgg/block9_box/conv_cls/weightssave/RestoreV2:26*
T0*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*'
_output_shapes
:?~*
validate_shape(
?
save/Assign_27Assign&ssd_300_vgg/block9_box/conv_loc/biasessave/RestoreV2:27*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_loc/biases*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
?
save/Assign_28Assign'ssd_300_vgg/block9_box/conv_loc/weightssave/RestoreV2:28*
validate_shape(*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*
use_locking(*
T0*'
_output_shapes
:?
?
save/Assign_29Assignssd_300_vgg/conv10_1/biasessave/RestoreV2:29*
validate_shape(*
T0*.
_class$
" loc:@ssd_300_vgg/conv10_1/biases*
use_locking(*
_output_shapes	
:?
?
save/Assign_30Assignssd_300_vgg/conv10_1/weightssave/RestoreV2:30*
use_locking(*(
_output_shapes
:??*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*
validate_shape(*
T0
?
save/Assign_31Assignssd_300_vgg/conv10_2/biasessave/RestoreV2:31*
use_locking(*
validate_shape(*.
_class$
" loc:@ssd_300_vgg/conv10_2/biases*
T0*
_output_shapes	
:?
?
save/Assign_32Assignssd_300_vgg/conv10_2/weightssave/RestoreV2:32*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*/
_class%
#!loc:@ssd_300_vgg/conv10_2/weights
?
save/Assign_33Assignssd_300_vgg/conv8_1/biasessave/RestoreV2:33*
validate_shape(*
T0*
use_locking(*-
_class#
!loc:@ssd_300_vgg/conv8_1/biases*
_output_shapes	
:?
?
save/Assign_34Assignssd_300_vgg/conv8_1/weightssave/RestoreV2:34*
validate_shape(*
use_locking(*
T0*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*(
_output_shapes
:??
?
save/Assign_35Assignssd_300_vgg/conv8_2/biasessave/RestoreV2:35*
validate_shape(*
_output_shapes	
:?*-
_class#
!loc:@ssd_300_vgg/conv8_2/biases*
T0*
use_locking(
?
save/Assign_36Assignssd_300_vgg/conv8_2/weightssave/RestoreV2:36*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights*(
_output_shapes
:??*
T0*
validate_shape(*
use_locking(
?
save/Assign_37Assignssd_300_vgg/conv9_1/biasessave/RestoreV2:37*-
_class#
!loc:@ssd_300_vgg/conv9_1/biases*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_38Assignssd_300_vgg/conv9_1/weightssave/RestoreV2:38*
T0*(
_output_shapes
:??*
use_locking(*
validate_shape(*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights
?
save/Assign_39Assignssd_300_vgg/conv9_2/biasessave/RestoreV2:39*-
_class#
!loc:@ssd_300_vgg/conv9_2/biases*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0
?
save/Assign_40Assignssd_300_vgg/conv9_2/weightssave/RestoreV2:40*
validate_shape(*
T0*
use_locking(*(
_output_shapes
:??*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights
?
save/Assign_41Assignssd_300_vgg/conv_init/biasessave/RestoreV2:41*/
_class%
#!loc:@ssd_300_vgg/conv_init/biases*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
?
save/Assign_42Assignssd_300_vgg/conv_init/weightssave/RestoreV2:42*
validate_shape(*
use_locking(*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*
T0*&
_output_shapes
:@
?
save/Assign_43Assignssd_300_vgg/conv_init_1/biasessave/RestoreV2:43*
use_locking(*
_output_shapes
:@*
validate_shape(*1
_class'
%#loc:@ssd_300_vgg/conv_init_1/biases*
T0
?
save/Assign_44Assignssd_300_vgg/conv_init_1/weightssave/RestoreV2:44*&
_output_shapes
:@@*
use_locking(*
T0*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights*
validate_shape(
?
save/Assign_45Assign)ssd_300_vgg/resblock0_0/batch_norm_0/betasave/RestoreV2:45*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_0/beta*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
?
save/Assign_46Assign*ssd_300_vgg/resblock0_0/batch_norm_0/gammasave/RestoreV2:46*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_0/gamma*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
?
save/Assign_47Assign0ssd_300_vgg/resblock0_0/batch_norm_0/moving_meansave/RestoreV2:47*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(*C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean
?
save/Assign_48Assign4ssd_300_vgg/resblock0_0/batch_norm_0/moving_variancesave/RestoreV2:48*
use_locking(*
validate_shape(*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance*
_output_shapes
:@*
T0
?
save/Assign_49Assign)ssd_300_vgg/resblock0_0/batch_norm_1/betasave/RestoreV2:49*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_1/beta*
use_locking(*
validate_shape(*
_output_shapes
:@
?
save/Assign_50Assign*ssd_300_vgg/resblock0_0/batch_norm_1/gammasave/RestoreV2:50*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_1/gamma
?
save/Assign_51Assign0ssd_300_vgg/resblock0_0/batch_norm_1/moving_meansave/RestoreV2:51*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean
?
save/Assign_52Assign4ssd_300_vgg/resblock0_0/batch_norm_1/moving_variancesave/RestoreV2:52*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(
?
save/Assign_53Assign%ssd_300_vgg/resblock0_0/conv_0/biasessave/RestoreV2:53*
use_locking(*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_0/biases*
T0*
_output_shapes
:@*
validate_shape(
?
save/Assign_54Assign&ssd_300_vgg/resblock0_0/conv_0/weightssave/RestoreV2:54*
T0*
use_locking(*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights*&
_output_shapes
:@@
?
save/Assign_55Assign%ssd_300_vgg/resblock0_0/conv_1/biasessave/RestoreV2:55*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_1/biases
?
save/Assign_56Assign&ssd_300_vgg/resblock0_0/conv_1/weightssave/RestoreV2:56*
use_locking(*&
_output_shapes
:@@*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*
validate_shape(
?
save/Assign_57Assign)ssd_300_vgg/resblock0_1/batch_norm_0/betasave/RestoreV2:57*
T0*
_output_shapes
:@*
validate_shape(*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_0/beta*
use_locking(
?
save/Assign_58Assign*ssd_300_vgg/resblock0_1/batch_norm_0/gammasave/RestoreV2:58*
_output_shapes
:@*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_0/gamma*
validate_shape(*
use_locking(
?
save/Assign_59Assign0ssd_300_vgg/resblock0_1/batch_norm_0/moving_meansave/RestoreV2:59*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean
?
save/Assign_60Assign4ssd_300_vgg/resblock0_1/batch_norm_0/moving_variancesave/RestoreV2:60*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance*
_output_shapes
:@*
validate_shape(*
use_locking(
?
save/Assign_61Assign)ssd_300_vgg/resblock0_1/batch_norm_1/betasave/RestoreV2:61*
use_locking(*
validate_shape(*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_1/beta*
_output_shapes
:@*
T0
?
save/Assign_62Assign*ssd_300_vgg/resblock0_1/batch_norm_1/gammasave/RestoreV2:62*
_output_shapes
:@*
validate_shape(*
use_locking(*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_1/gamma
?
save/Assign_63Assign0ssd_300_vgg/resblock0_1/batch_norm_1/moving_meansave/RestoreV2:63*
validate_shape(*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean*
use_locking(*
T0
?
save/Assign_64Assign4ssd_300_vgg/resblock0_1/batch_norm_1/moving_variancesave/RestoreV2:64*
validate_shape(*
use_locking(*
_output_shapes
:@*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance*
T0
?
save/Assign_65Assign%ssd_300_vgg/resblock0_1/conv_0/biasessave/RestoreV2:65*
use_locking(*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_0/biases*
_output_shapes
:@*
T0*
validate_shape(
?
save/Assign_66Assign&ssd_300_vgg/resblock0_1/conv_0/weightssave/RestoreV2:66*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights*
use_locking(*&
_output_shapes
:@@*
validate_shape(
?
save/Assign_67Assign%ssd_300_vgg/resblock0_1/conv_1/biasessave/RestoreV2:67*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_1/biases
?
save/Assign_68Assign&ssd_300_vgg/resblock0_1/conv_1/weightssave/RestoreV2:68*&
_output_shapes
:@@*
T0*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*
validate_shape(
?
save/Assign_69Assign)ssd_300_vgg/resblock1_0/batch_norm_0/betasave/RestoreV2:69*
use_locking(*
_output_shapes
:@*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_0/beta*
T0*
validate_shape(
?
save/Assign_70Assign*ssd_300_vgg/resblock1_0/batch_norm_0/gammasave/RestoreV2:70*
validate_shape(*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_0/gamma*
_output_shapes
:@*
use_locking(
?
save/Assign_71Assign0ssd_300_vgg/resblock1_0/batch_norm_0/moving_meansave/RestoreV2:71*
use_locking(*
_output_shapes
:@*
T0*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean*
validate_shape(
?
save/Assign_72Assign4ssd_300_vgg/resblock1_0/batch_norm_0/moving_variancesave/RestoreV2:72*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
?
save/Assign_73Assign)ssd_300_vgg/resblock1_0/batch_norm_1/betasave/RestoreV2:73*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_1/beta
?
save/Assign_74Assign*ssd_300_vgg/resblock1_0/batch_norm_1/gammasave/RestoreV2:74*
T0*
_output_shapes	
:?*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_1/gamma*
use_locking(
?
save/Assign_75Assign0ssd_300_vgg/resblock1_0/batch_norm_1/moving_meansave/RestoreV2:75*
validate_shape(*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean*
T0*
use_locking(
?
save/Assign_76Assign4ssd_300_vgg/resblock1_0/batch_norm_1/moving_variancesave/RestoreV2:76*
use_locking(*
T0*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance*
validate_shape(
?
save/Assign_77Assign%ssd_300_vgg/resblock1_0/conv_0/biasessave/RestoreV2:77*
_output_shapes	
:?*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_0/biases*
T0*
use_locking(
?
save/Assign_78Assign&ssd_300_vgg/resblock1_0/conv_0/weightssave/RestoreV2:78*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*'
_output_shapes
:@?*
use_locking(*
T0
?
save/Assign_79Assign%ssd_300_vgg/resblock1_0/conv_1/biasessave/RestoreV2:79*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_1/biases*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save/Assign_80Assign&ssd_300_vgg/resblock1_0/conv_1/weightssave/RestoreV2:80*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights*(
_output_shapes
:??*
validate_shape(*
T0
?
save/Assign_81Assign(ssd_300_vgg/resblock1_0/conv_init/biasessave/RestoreV2:81*
use_locking(*
T0*;
_class1
/-loc:@ssd_300_vgg/resblock1_0/conv_init/biases*
_output_shapes	
:?*
validate_shape(
?
save/Assign_82Assign)ssd_300_vgg/resblock1_0/conv_init/weightssave/RestoreV2:82*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights*
use_locking(*'
_output_shapes
:@?*
validate_shape(*
T0
?
save/Assign_83Assign)ssd_300_vgg/resblock1_1/batch_norm_0/betasave/RestoreV2:83*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_0/beta*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_84Assign*ssd_300_vgg/resblock1_1/batch_norm_0/gammasave/RestoreV2:84*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_0/gamma*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save/Assign_85Assign0ssd_300_vgg/resblock1_1/batch_norm_0/moving_meansave/RestoreV2:85*
T0*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_86Assign4ssd_300_vgg/resblock1_1/batch_norm_0/moving_variancesave/RestoreV2:86*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(
?
save/Assign_87Assign)ssd_300_vgg/resblock1_1/batch_norm_1/betasave/RestoreV2:87*
T0*
validate_shape(*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_1/beta*
use_locking(*
_output_shapes	
:?
?
save/Assign_88Assign*ssd_300_vgg/resblock1_1/batch_norm_1/gammasave/RestoreV2:88*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_1/gamma
?
save/Assign_89Assign0ssd_300_vgg/resblock1_1/batch_norm_1/moving_meansave/RestoreV2:89*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean
?
save/Assign_90Assign4ssd_300_vgg/resblock1_1/batch_norm_1/moving_variancesave/RestoreV2:90*
T0*
validate_shape(*
use_locking(*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance*
_output_shapes	
:?
?
save/Assign_91Assign%ssd_300_vgg/resblock1_1/conv_0/biasessave/RestoreV2:91*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_0/biases
?
save/Assign_92Assign&ssd_300_vgg/resblock1_1/conv_0/weightssave/RestoreV2:92*
use_locking(*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights*
validate_shape(*
T0
?
save/Assign_93Assign%ssd_300_vgg/resblock1_1/conv_1/biasessave/RestoreV2:93*
use_locking(*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_1/biases*
_output_shapes	
:?*
validate_shape(*
T0
?
save/Assign_94Assign&ssd_300_vgg/resblock1_1/conv_1/weightssave/RestoreV2:94*
validate_shape(*
T0*
use_locking(*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights
?
save/Assign_95Assign)ssd_300_vgg/resblock2_0/batch_norm_0/betasave/RestoreV2:95*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_0/beta*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(
?
save/Assign_96Assign*ssd_300_vgg/resblock2_0/batch_norm_0/gammasave/RestoreV2:96*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_0/gamma*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
save/Assign_97Assign0ssd_300_vgg/resblock2_0/batch_norm_0/moving_meansave/RestoreV2:97*
T0*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean*
validate_shape(*
use_locking(
?
save/Assign_98Assign4ssd_300_vgg/resblock2_0/batch_norm_0/moving_variancesave/RestoreV2:98*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save/Assign_99Assign)ssd_300_vgg/resblock2_0/batch_norm_1/betasave/RestoreV2:99*
T0*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_1/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_100Assign*ssd_300_vgg/resblock2_0/batch_norm_1/gammasave/RestoreV2:100*
use_locking(*
validate_shape(*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_1/gamma*
T0
?
save/Assign_101Assign0ssd_300_vgg/resblock2_0/batch_norm_1/moving_meansave/RestoreV2:101*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(
?
save/Assign_102Assign4ssd_300_vgg/resblock2_0/batch_norm_1/moving_variancesave/RestoreV2:102*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance
?
save/Assign_103Assign%ssd_300_vgg/resblock2_0/conv_0/biasessave/RestoreV2:103*
_output_shapes	
:?*
use_locking(*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_0/biases*
T0*
validate_shape(
?
save/Assign_104Assign&ssd_300_vgg/resblock2_0/conv_0/weightssave/RestoreV2:104*
use_locking(*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*
T0*(
_output_shapes
:??
?
save/Assign_105Assign%ssd_300_vgg/resblock2_0/conv_1/biasessave/RestoreV2:105*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_1/biases*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0
?
save/Assign_106Assign&ssd_300_vgg/resblock2_0/conv_1/weightssave/RestoreV2:106*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*
use_locking(*
validate_shape(*(
_output_shapes
:??
?
save/Assign_107Assign(ssd_300_vgg/resblock2_0/conv_init/biasessave/RestoreV2:107*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?*;
_class1
/-loc:@ssd_300_vgg/resblock2_0/conv_init/biases
?
save/Assign_108Assign)ssd_300_vgg/resblock2_0/conv_init/weightssave/RestoreV2:108*
use_locking(*(
_output_shapes
:??*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*
validate_shape(*
T0
?
save/Assign_109Assign)ssd_300_vgg/resblock2_1/batch_norm_0/betasave/RestoreV2:109*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_0/beta*
validate_shape(*
_output_shapes	
:?*
T0
?
save/Assign_110Assign*ssd_300_vgg/resblock2_1/batch_norm_0/gammasave/RestoreV2:110*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_0/gamma*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save/Assign_111Assign0ssd_300_vgg/resblock2_1/batch_norm_0/moving_meansave/RestoreV2:111*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(
?
save/Assign_112Assign4ssd_300_vgg/resblock2_1/batch_norm_0/moving_variancesave/RestoreV2:112*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance
?
save/Assign_113Assign)ssd_300_vgg/resblock2_1/batch_norm_1/betasave/RestoreV2:113*
validate_shape(*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_1/beta*
_output_shapes	
:?*
T0
?
save/Assign_114Assign*ssd_300_vgg/resblock2_1/batch_norm_1/gammasave/RestoreV2:114*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_1/gamma*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0
?
save/Assign_115Assign0ssd_300_vgg/resblock2_1/batch_norm_1/moving_meansave/RestoreV2:115*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?
?
save/Assign_116Assign4ssd_300_vgg/resblock2_1/batch_norm_1/moving_variancesave/RestoreV2:116*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance
?
save/Assign_117Assign%ssd_300_vgg/resblock2_1/conv_0/biasessave/RestoreV2:117*
use_locking(*
validate_shape(*
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_0/biases*
T0
?
save/Assign_118Assign&ssd_300_vgg/resblock2_1/conv_0/weightssave/RestoreV2:118*(
_output_shapes
:??*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights*
validate_shape(*
T0
?
save/Assign_119Assign%ssd_300_vgg/resblock2_1/conv_1/biasessave/RestoreV2:119*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_1/biases
?
save/Assign_120Assign&ssd_300_vgg/resblock2_1/conv_1/weightssave/RestoreV2:120*
T0*(
_output_shapes
:??*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights*
validate_shape(*
use_locking(
?
save/Assign_121Assign*ssd_300_vgg/resblock_3_0/batch_norm_0/betasave/RestoreV2:121*
_output_shapes	
:?*
T0*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/beta*
use_locking(
?
save/Assign_122Assign+ssd_300_vgg/resblock_3_0/batch_norm_0/gammasave/RestoreV2:122*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/gamma
?
save/Assign_123Assign1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_meansave/RestoreV2:123*
_output_shapes	
:?*
T0*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean*
validate_shape(*
use_locking(
?
save/Assign_124Assign5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variancesave/RestoreV2:124*
T0*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save/Assign_125Assign*ssd_300_vgg/resblock_3_0/batch_norm_1/betasave/RestoreV2:125*
validate_shape(*
T0*
use_locking(*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/beta*
_output_shapes	
:?
?
save/Assign_126Assign+ssd_300_vgg/resblock_3_0/batch_norm_1/gammasave/RestoreV2:126*
validate_shape(*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/gamma*
T0*
use_locking(*
_output_shapes	
:?
?
save/Assign_127Assign1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_meansave/RestoreV2:127*
T0*
_output_shapes	
:?*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean*
use_locking(*
validate_shape(
?
save/Assign_128Assign5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variancesave/RestoreV2:128*
validate_shape(*
_output_shapes	
:?*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance*
use_locking(*
T0
?
save/Assign_129Assign&ssd_300_vgg/resblock_3_0/conv_0/biasessave/RestoreV2:129*
T0*
_output_shapes	
:?*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_0/biases*
validate_shape(
?
save/Assign_130Assign'ssd_300_vgg/resblock_3_0/conv_0/weightssave/RestoreV2:130*
T0*
validate_shape(*
use_locking(*(
_output_shapes
:??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights
?
save/Assign_131Assign&ssd_300_vgg/resblock_3_0/conv_1/biasessave/RestoreV2:131*
validate_shape(*
_output_shapes	
:?*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_1/biases*
T0
?
save/Assign_132Assign'ssd_300_vgg/resblock_3_0/conv_1/weightssave/RestoreV2:132*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0
?
save/Assign_133Assign)ssd_300_vgg/resblock_3_0/conv_init/biasessave/RestoreV2:133*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock_3_0/conv_init/biases
?
save/Assign_134Assign*ssd_300_vgg/resblock_3_0/conv_init/weightssave/RestoreV2:134*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights*
validate_shape(*
T0*
use_locking(*(
_output_shapes
:??
?
save/Assign_135Assign*ssd_300_vgg/resblock_3_1/batch_norm_0/betasave/RestoreV2:135*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/beta*
T0*
validate_shape(*
use_locking(
?
save/Assign_136Assign+ssd_300_vgg/resblock_3_1/batch_norm_0/gammasave/RestoreV2:136*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/gamma*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0
?
save/Assign_137Assign1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_meansave/RestoreV2:137*
validate_shape(*
use_locking(*
_output_shapes	
:?*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean*
T0
?
save/Assign_138Assign5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variancesave/RestoreV2:138*
validate_shape(*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance*
T0*
use_locking(*
_output_shapes	
:?
?
save/Assign_139Assign*ssd_300_vgg/resblock_3_1/batch_norm_1/betasave/RestoreV2:139*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/beta*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?
?
save/Assign_140Assign+ssd_300_vgg/resblock_3_1/batch_norm_1/gammasave/RestoreV2:140*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/gamma*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
?
save/Assign_141Assign1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_meansave/RestoreV2:141*
validate_shape(*
_output_shapes	
:?*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean*
T0*
use_locking(
?
save/Assign_142Assign5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variancesave/RestoreV2:142*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance
?
save/Assign_143Assign&ssd_300_vgg/resblock_3_1/conv_0/biasessave/RestoreV2:143*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_0/biases*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save/Assign_144Assign'ssd_300_vgg/resblock_3_1/conv_0/weightssave/RestoreV2:144*
T0*
validate_shape(*
use_locking(*(
_output_shapes
:??*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights
?
save/Assign_145Assign&ssd_300_vgg/resblock_3_1/conv_1/biasessave/RestoreV2:145*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_1/biases
?
save/Assign_146Assign'ssd_300_vgg/resblock_3_1/conv_1/weightssave/RestoreV2:146*(
_output_shapes
:??*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights*
validate_shape(*
T0
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_11^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_12^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_129^save/Assign_13^save/Assign_130^save/Assign_131^save/Assign_132^save/Assign_133^save/Assign_134^save/Assign_135^save/Assign_136^save/Assign_137^save/Assign_138^save/Assign_139^save/Assign_14^save/Assign_140^save/Assign_141^save/Assign_142^save/Assign_143^save/Assign_144^save/Assign_145^save/Assign_146^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
[
save_1/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
_output_shapes
: *
dtype0
?
save_1/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_0dfec2b1cd4e432fb3f18f84396e9bdf/part*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_1/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
?1
save_1/SaveV2/tensor_namesConst*
_output_shapes	
:?*
dtype0*?0
value?0B?0?Bssd_300_vgg/batch_norm_00/betaBssd_300_vgg/batch_norm_00/gammaB%ssd_300_vgg/batch_norm_00/moving_meanB)ssd_300_vgg/batch_norm_00/moving_varianceB'ssd_300_vgg/block10_box/conv_cls/biasesB(ssd_300_vgg/block10_box/conv_cls/weightsB'ssd_300_vgg/block10_box/conv_loc/biasesB(ssd_300_vgg/block10_box/conv_loc/weightsB'ssd_300_vgg/block11_box/conv_cls/biasesB(ssd_300_vgg/block11_box/conv_cls/weightsB'ssd_300_vgg/block11_box/conv_loc/biasesB(ssd_300_vgg/block11_box/conv_loc/weightsB,ssd_300_vgg/block4_box/L2Normalization/gammaB&ssd_300_vgg/block4_box/conv_cls/biasesB'ssd_300_vgg/block4_box/conv_cls/weightsB&ssd_300_vgg/block4_box/conv_loc/biasesB'ssd_300_vgg/block4_box/conv_loc/weightsB&ssd_300_vgg/block7_box/conv_cls/biasesB'ssd_300_vgg/block7_box/conv_cls/weightsB&ssd_300_vgg/block7_box/conv_loc/biasesB'ssd_300_vgg/block7_box/conv_loc/weightsB&ssd_300_vgg/block8_box/conv_cls/biasesB'ssd_300_vgg/block8_box/conv_cls/weightsB&ssd_300_vgg/block8_box/conv_loc/biasesB'ssd_300_vgg/block8_box/conv_loc/weightsB&ssd_300_vgg/block9_box/conv_cls/biasesB'ssd_300_vgg/block9_box/conv_cls/weightsB&ssd_300_vgg/block9_box/conv_loc/biasesB'ssd_300_vgg/block9_box/conv_loc/weightsBssd_300_vgg/conv10_1/biasesBssd_300_vgg/conv10_1/weightsBssd_300_vgg/conv10_2/biasesBssd_300_vgg/conv10_2/weightsBssd_300_vgg/conv8_1/biasesBssd_300_vgg/conv8_1/weightsBssd_300_vgg/conv8_2/biasesBssd_300_vgg/conv8_2/weightsBssd_300_vgg/conv9_1/biasesBssd_300_vgg/conv9_1/weightsBssd_300_vgg/conv9_2/biasesBssd_300_vgg/conv9_2/weightsBssd_300_vgg/conv_init/biasesBssd_300_vgg/conv_init/weightsBssd_300_vgg/conv_init_1/biasesBssd_300_vgg/conv_init_1/weightsB)ssd_300_vgg/resblock0_0/batch_norm_0/betaB*ssd_300_vgg/resblock0_0/batch_norm_0/gammaB0ssd_300_vgg/resblock0_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock0_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock0_0/batch_norm_1/betaB*ssd_300_vgg/resblock0_0/batch_norm_1/gammaB0ssd_300_vgg/resblock0_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock0_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock0_0/conv_0/biasesB&ssd_300_vgg/resblock0_0/conv_0/weightsB%ssd_300_vgg/resblock0_0/conv_1/biasesB&ssd_300_vgg/resblock0_0/conv_1/weightsB)ssd_300_vgg/resblock0_1/batch_norm_0/betaB*ssd_300_vgg/resblock0_1/batch_norm_0/gammaB0ssd_300_vgg/resblock0_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock0_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock0_1/batch_norm_1/betaB*ssd_300_vgg/resblock0_1/batch_norm_1/gammaB0ssd_300_vgg/resblock0_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock0_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock0_1/conv_0/biasesB&ssd_300_vgg/resblock0_1/conv_0/weightsB%ssd_300_vgg/resblock0_1/conv_1/biasesB&ssd_300_vgg/resblock0_1/conv_1/weightsB)ssd_300_vgg/resblock1_0/batch_norm_0/betaB*ssd_300_vgg/resblock1_0/batch_norm_0/gammaB0ssd_300_vgg/resblock1_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock1_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock1_0/batch_norm_1/betaB*ssd_300_vgg/resblock1_0/batch_norm_1/gammaB0ssd_300_vgg/resblock1_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock1_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock1_0/conv_0/biasesB&ssd_300_vgg/resblock1_0/conv_0/weightsB%ssd_300_vgg/resblock1_0/conv_1/biasesB&ssd_300_vgg/resblock1_0/conv_1/weightsB(ssd_300_vgg/resblock1_0/conv_init/biasesB)ssd_300_vgg/resblock1_0/conv_init/weightsB)ssd_300_vgg/resblock1_1/batch_norm_0/betaB*ssd_300_vgg/resblock1_1/batch_norm_0/gammaB0ssd_300_vgg/resblock1_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock1_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock1_1/batch_norm_1/betaB*ssd_300_vgg/resblock1_1/batch_norm_1/gammaB0ssd_300_vgg/resblock1_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock1_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock1_1/conv_0/biasesB&ssd_300_vgg/resblock1_1/conv_0/weightsB%ssd_300_vgg/resblock1_1/conv_1/biasesB&ssd_300_vgg/resblock1_1/conv_1/weightsB)ssd_300_vgg/resblock2_0/batch_norm_0/betaB*ssd_300_vgg/resblock2_0/batch_norm_0/gammaB0ssd_300_vgg/resblock2_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock2_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock2_0/batch_norm_1/betaB*ssd_300_vgg/resblock2_0/batch_norm_1/gammaB0ssd_300_vgg/resblock2_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock2_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock2_0/conv_0/biasesB&ssd_300_vgg/resblock2_0/conv_0/weightsB%ssd_300_vgg/resblock2_0/conv_1/biasesB&ssd_300_vgg/resblock2_0/conv_1/weightsB(ssd_300_vgg/resblock2_0/conv_init/biasesB)ssd_300_vgg/resblock2_0/conv_init/weightsB)ssd_300_vgg/resblock2_1/batch_norm_0/betaB*ssd_300_vgg/resblock2_1/batch_norm_0/gammaB0ssd_300_vgg/resblock2_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock2_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock2_1/batch_norm_1/betaB*ssd_300_vgg/resblock2_1/batch_norm_1/gammaB0ssd_300_vgg/resblock2_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock2_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock2_1/conv_0/biasesB&ssd_300_vgg/resblock2_1/conv_0/weightsB%ssd_300_vgg/resblock2_1/conv_1/biasesB&ssd_300_vgg/resblock2_1/conv_1/weightsB*ssd_300_vgg/resblock_3_0/batch_norm_0/betaB+ssd_300_vgg/resblock_3_0/batch_norm_0/gammaB1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_meanB5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_varianceB*ssd_300_vgg/resblock_3_0/batch_norm_1/betaB+ssd_300_vgg/resblock_3_0/batch_norm_1/gammaB1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_meanB5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_varianceB&ssd_300_vgg/resblock_3_0/conv_0/biasesB'ssd_300_vgg/resblock_3_0/conv_0/weightsB&ssd_300_vgg/resblock_3_0/conv_1/biasesB'ssd_300_vgg/resblock_3_0/conv_1/weightsB)ssd_300_vgg/resblock_3_0/conv_init/biasesB*ssd_300_vgg/resblock_3_0/conv_init/weightsB*ssd_300_vgg/resblock_3_1/batch_norm_0/betaB+ssd_300_vgg/resblock_3_1/batch_norm_0/gammaB1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_meanB5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_varianceB*ssd_300_vgg/resblock_3_1/batch_norm_1/betaB+ssd_300_vgg/resblock_3_1/batch_norm_1/gammaB1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_meanB5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_varianceB&ssd_300_vgg/resblock_3_1/conv_0/biasesB'ssd_300_vgg/resblock_3_1/conv_0/weightsB&ssd_300_vgg/resblock_3_1/conv_1/biasesB'ssd_300_vgg/resblock_3_1/conv_1/weights
?
save_1/SaveV2/shape_and_slicesConst*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes	
:?*
dtype0
?2
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesssd_300_vgg/batch_norm_00/betassd_300_vgg/batch_norm_00/gamma%ssd_300_vgg/batch_norm_00/moving_mean)ssd_300_vgg/batch_norm_00/moving_variance'ssd_300_vgg/block10_box/conv_cls/biases(ssd_300_vgg/block10_box/conv_cls/weights'ssd_300_vgg/block10_box/conv_loc/biases(ssd_300_vgg/block10_box/conv_loc/weights'ssd_300_vgg/block11_box/conv_cls/biases(ssd_300_vgg/block11_box/conv_cls/weights'ssd_300_vgg/block11_box/conv_loc/biases(ssd_300_vgg/block11_box/conv_loc/weights,ssd_300_vgg/block4_box/L2Normalization/gamma&ssd_300_vgg/block4_box/conv_cls/biases'ssd_300_vgg/block4_box/conv_cls/weights&ssd_300_vgg/block4_box/conv_loc/biases'ssd_300_vgg/block4_box/conv_loc/weights&ssd_300_vgg/block7_box/conv_cls/biases'ssd_300_vgg/block7_box/conv_cls/weights&ssd_300_vgg/block7_box/conv_loc/biases'ssd_300_vgg/block7_box/conv_loc/weights&ssd_300_vgg/block8_box/conv_cls/biases'ssd_300_vgg/block8_box/conv_cls/weights&ssd_300_vgg/block8_box/conv_loc/biases'ssd_300_vgg/block8_box/conv_loc/weights&ssd_300_vgg/block9_box/conv_cls/biases'ssd_300_vgg/block9_box/conv_cls/weights&ssd_300_vgg/block9_box/conv_loc/biases'ssd_300_vgg/block9_box/conv_loc/weightsssd_300_vgg/conv10_1/biasesssd_300_vgg/conv10_1/weightsssd_300_vgg/conv10_2/biasesssd_300_vgg/conv10_2/weightsssd_300_vgg/conv8_1/biasesssd_300_vgg/conv8_1/weightsssd_300_vgg/conv8_2/biasesssd_300_vgg/conv8_2/weightsssd_300_vgg/conv9_1/biasesssd_300_vgg/conv9_1/weightsssd_300_vgg/conv9_2/biasesssd_300_vgg/conv9_2/weightsssd_300_vgg/conv_init/biasesssd_300_vgg/conv_init/weightsssd_300_vgg/conv_init_1/biasesssd_300_vgg/conv_init_1/weights)ssd_300_vgg/resblock0_0/batch_norm_0/beta*ssd_300_vgg/resblock0_0/batch_norm_0/gamma0ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean4ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance)ssd_300_vgg/resblock0_0/batch_norm_1/beta*ssd_300_vgg/resblock0_0/batch_norm_1/gamma0ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean4ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance%ssd_300_vgg/resblock0_0/conv_0/biases&ssd_300_vgg/resblock0_0/conv_0/weights%ssd_300_vgg/resblock0_0/conv_1/biases&ssd_300_vgg/resblock0_0/conv_1/weights)ssd_300_vgg/resblock0_1/batch_norm_0/beta*ssd_300_vgg/resblock0_1/batch_norm_0/gamma0ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean4ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance)ssd_300_vgg/resblock0_1/batch_norm_1/beta*ssd_300_vgg/resblock0_1/batch_norm_1/gamma0ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean4ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance%ssd_300_vgg/resblock0_1/conv_0/biases&ssd_300_vgg/resblock0_1/conv_0/weights%ssd_300_vgg/resblock0_1/conv_1/biases&ssd_300_vgg/resblock0_1/conv_1/weights)ssd_300_vgg/resblock1_0/batch_norm_0/beta*ssd_300_vgg/resblock1_0/batch_norm_0/gamma0ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean4ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance)ssd_300_vgg/resblock1_0/batch_norm_1/beta*ssd_300_vgg/resblock1_0/batch_norm_1/gamma0ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean4ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance%ssd_300_vgg/resblock1_0/conv_0/biases&ssd_300_vgg/resblock1_0/conv_0/weights%ssd_300_vgg/resblock1_0/conv_1/biases&ssd_300_vgg/resblock1_0/conv_1/weights(ssd_300_vgg/resblock1_0/conv_init/biases)ssd_300_vgg/resblock1_0/conv_init/weights)ssd_300_vgg/resblock1_1/batch_norm_0/beta*ssd_300_vgg/resblock1_1/batch_norm_0/gamma0ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean4ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance)ssd_300_vgg/resblock1_1/batch_norm_1/beta*ssd_300_vgg/resblock1_1/batch_norm_1/gamma0ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean4ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance%ssd_300_vgg/resblock1_1/conv_0/biases&ssd_300_vgg/resblock1_1/conv_0/weights%ssd_300_vgg/resblock1_1/conv_1/biases&ssd_300_vgg/resblock1_1/conv_1/weights)ssd_300_vgg/resblock2_0/batch_norm_0/beta*ssd_300_vgg/resblock2_0/batch_norm_0/gamma0ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean4ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance)ssd_300_vgg/resblock2_0/batch_norm_1/beta*ssd_300_vgg/resblock2_0/batch_norm_1/gamma0ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean4ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance%ssd_300_vgg/resblock2_0/conv_0/biases&ssd_300_vgg/resblock2_0/conv_0/weights%ssd_300_vgg/resblock2_0/conv_1/biases&ssd_300_vgg/resblock2_0/conv_1/weights(ssd_300_vgg/resblock2_0/conv_init/biases)ssd_300_vgg/resblock2_0/conv_init/weights)ssd_300_vgg/resblock2_1/batch_norm_0/beta*ssd_300_vgg/resblock2_1/batch_norm_0/gamma0ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean4ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance)ssd_300_vgg/resblock2_1/batch_norm_1/beta*ssd_300_vgg/resblock2_1/batch_norm_1/gamma0ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean4ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance%ssd_300_vgg/resblock2_1/conv_0/biases&ssd_300_vgg/resblock2_1/conv_0/weights%ssd_300_vgg/resblock2_1/conv_1/biases&ssd_300_vgg/resblock2_1/conv_1/weights*ssd_300_vgg/resblock_3_0/batch_norm_0/beta+ssd_300_vgg/resblock_3_0/batch_norm_0/gamma1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance*ssd_300_vgg/resblock_3_0/batch_norm_1/beta+ssd_300_vgg/resblock_3_0/batch_norm_1/gamma1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance&ssd_300_vgg/resblock_3_0/conv_0/biases'ssd_300_vgg/resblock_3_0/conv_0/weights&ssd_300_vgg/resblock_3_0/conv_1/biases'ssd_300_vgg/resblock_3_0/conv_1/weights)ssd_300_vgg/resblock_3_0/conv_init/biases*ssd_300_vgg/resblock_3_0/conv_init/weights*ssd_300_vgg/resblock_3_1/batch_norm_0/beta+ssd_300_vgg/resblock_3_1/batch_norm_0/gamma1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance*ssd_300_vgg/resblock_3_1/batch_norm_1/beta+ssd_300_vgg/resblock_3_1/batch_norm_1/gamma1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance&ssd_300_vgg/resblock_3_1/conv_0/biases'ssd_300_vgg/resblock_3_1/conv_0/weights&ssd_300_vgg/resblock_3_1/conv_1/biases'ssd_300_vgg/resblock_3_1/conv_1/weights*?
dtypes?
?2?
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename*
T0
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
N*
_output_shapes
:*
T0*

axis 
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
?1
save_1/RestoreV2/tensor_namesConst*
_output_shapes	
:?*?0
value?0B?0?Bssd_300_vgg/batch_norm_00/betaBssd_300_vgg/batch_norm_00/gammaB%ssd_300_vgg/batch_norm_00/moving_meanB)ssd_300_vgg/batch_norm_00/moving_varianceB'ssd_300_vgg/block10_box/conv_cls/biasesB(ssd_300_vgg/block10_box/conv_cls/weightsB'ssd_300_vgg/block10_box/conv_loc/biasesB(ssd_300_vgg/block10_box/conv_loc/weightsB'ssd_300_vgg/block11_box/conv_cls/biasesB(ssd_300_vgg/block11_box/conv_cls/weightsB'ssd_300_vgg/block11_box/conv_loc/biasesB(ssd_300_vgg/block11_box/conv_loc/weightsB,ssd_300_vgg/block4_box/L2Normalization/gammaB&ssd_300_vgg/block4_box/conv_cls/biasesB'ssd_300_vgg/block4_box/conv_cls/weightsB&ssd_300_vgg/block4_box/conv_loc/biasesB'ssd_300_vgg/block4_box/conv_loc/weightsB&ssd_300_vgg/block7_box/conv_cls/biasesB'ssd_300_vgg/block7_box/conv_cls/weightsB&ssd_300_vgg/block7_box/conv_loc/biasesB'ssd_300_vgg/block7_box/conv_loc/weightsB&ssd_300_vgg/block8_box/conv_cls/biasesB'ssd_300_vgg/block8_box/conv_cls/weightsB&ssd_300_vgg/block8_box/conv_loc/biasesB'ssd_300_vgg/block8_box/conv_loc/weightsB&ssd_300_vgg/block9_box/conv_cls/biasesB'ssd_300_vgg/block9_box/conv_cls/weightsB&ssd_300_vgg/block9_box/conv_loc/biasesB'ssd_300_vgg/block9_box/conv_loc/weightsBssd_300_vgg/conv10_1/biasesBssd_300_vgg/conv10_1/weightsBssd_300_vgg/conv10_2/biasesBssd_300_vgg/conv10_2/weightsBssd_300_vgg/conv8_1/biasesBssd_300_vgg/conv8_1/weightsBssd_300_vgg/conv8_2/biasesBssd_300_vgg/conv8_2/weightsBssd_300_vgg/conv9_1/biasesBssd_300_vgg/conv9_1/weightsBssd_300_vgg/conv9_2/biasesBssd_300_vgg/conv9_2/weightsBssd_300_vgg/conv_init/biasesBssd_300_vgg/conv_init/weightsBssd_300_vgg/conv_init_1/biasesBssd_300_vgg/conv_init_1/weightsB)ssd_300_vgg/resblock0_0/batch_norm_0/betaB*ssd_300_vgg/resblock0_0/batch_norm_0/gammaB0ssd_300_vgg/resblock0_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock0_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock0_0/batch_norm_1/betaB*ssd_300_vgg/resblock0_0/batch_norm_1/gammaB0ssd_300_vgg/resblock0_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock0_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock0_0/conv_0/biasesB&ssd_300_vgg/resblock0_0/conv_0/weightsB%ssd_300_vgg/resblock0_0/conv_1/biasesB&ssd_300_vgg/resblock0_0/conv_1/weightsB)ssd_300_vgg/resblock0_1/batch_norm_0/betaB*ssd_300_vgg/resblock0_1/batch_norm_0/gammaB0ssd_300_vgg/resblock0_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock0_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock0_1/batch_norm_1/betaB*ssd_300_vgg/resblock0_1/batch_norm_1/gammaB0ssd_300_vgg/resblock0_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock0_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock0_1/conv_0/biasesB&ssd_300_vgg/resblock0_1/conv_0/weightsB%ssd_300_vgg/resblock0_1/conv_1/biasesB&ssd_300_vgg/resblock0_1/conv_1/weightsB)ssd_300_vgg/resblock1_0/batch_norm_0/betaB*ssd_300_vgg/resblock1_0/batch_norm_0/gammaB0ssd_300_vgg/resblock1_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock1_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock1_0/batch_norm_1/betaB*ssd_300_vgg/resblock1_0/batch_norm_1/gammaB0ssd_300_vgg/resblock1_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock1_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock1_0/conv_0/biasesB&ssd_300_vgg/resblock1_0/conv_0/weightsB%ssd_300_vgg/resblock1_0/conv_1/biasesB&ssd_300_vgg/resblock1_0/conv_1/weightsB(ssd_300_vgg/resblock1_0/conv_init/biasesB)ssd_300_vgg/resblock1_0/conv_init/weightsB)ssd_300_vgg/resblock1_1/batch_norm_0/betaB*ssd_300_vgg/resblock1_1/batch_norm_0/gammaB0ssd_300_vgg/resblock1_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock1_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock1_1/batch_norm_1/betaB*ssd_300_vgg/resblock1_1/batch_norm_1/gammaB0ssd_300_vgg/resblock1_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock1_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock1_1/conv_0/biasesB&ssd_300_vgg/resblock1_1/conv_0/weightsB%ssd_300_vgg/resblock1_1/conv_1/biasesB&ssd_300_vgg/resblock1_1/conv_1/weightsB)ssd_300_vgg/resblock2_0/batch_norm_0/betaB*ssd_300_vgg/resblock2_0/batch_norm_0/gammaB0ssd_300_vgg/resblock2_0/batch_norm_0/moving_meanB4ssd_300_vgg/resblock2_0/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock2_0/batch_norm_1/betaB*ssd_300_vgg/resblock2_0/batch_norm_1/gammaB0ssd_300_vgg/resblock2_0/batch_norm_1/moving_meanB4ssd_300_vgg/resblock2_0/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock2_0/conv_0/biasesB&ssd_300_vgg/resblock2_0/conv_0/weightsB%ssd_300_vgg/resblock2_0/conv_1/biasesB&ssd_300_vgg/resblock2_0/conv_1/weightsB(ssd_300_vgg/resblock2_0/conv_init/biasesB)ssd_300_vgg/resblock2_0/conv_init/weightsB)ssd_300_vgg/resblock2_1/batch_norm_0/betaB*ssd_300_vgg/resblock2_1/batch_norm_0/gammaB0ssd_300_vgg/resblock2_1/batch_norm_0/moving_meanB4ssd_300_vgg/resblock2_1/batch_norm_0/moving_varianceB)ssd_300_vgg/resblock2_1/batch_norm_1/betaB*ssd_300_vgg/resblock2_1/batch_norm_1/gammaB0ssd_300_vgg/resblock2_1/batch_norm_1/moving_meanB4ssd_300_vgg/resblock2_1/batch_norm_1/moving_varianceB%ssd_300_vgg/resblock2_1/conv_0/biasesB&ssd_300_vgg/resblock2_1/conv_0/weightsB%ssd_300_vgg/resblock2_1/conv_1/biasesB&ssd_300_vgg/resblock2_1/conv_1/weightsB*ssd_300_vgg/resblock_3_0/batch_norm_0/betaB+ssd_300_vgg/resblock_3_0/batch_norm_0/gammaB1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_meanB5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_varianceB*ssd_300_vgg/resblock_3_0/batch_norm_1/betaB+ssd_300_vgg/resblock_3_0/batch_norm_1/gammaB1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_meanB5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_varianceB&ssd_300_vgg/resblock_3_0/conv_0/biasesB'ssd_300_vgg/resblock_3_0/conv_0/weightsB&ssd_300_vgg/resblock_3_0/conv_1/biasesB'ssd_300_vgg/resblock_3_0/conv_1/weightsB)ssd_300_vgg/resblock_3_0/conv_init/biasesB*ssd_300_vgg/resblock_3_0/conv_init/weightsB*ssd_300_vgg/resblock_3_1/batch_norm_0/betaB+ssd_300_vgg/resblock_3_1/batch_norm_0/gammaB1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_meanB5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_varianceB*ssd_300_vgg/resblock_3_1/batch_norm_1/betaB+ssd_300_vgg/resblock_3_1/batch_norm_1/gammaB1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_meanB5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_varianceB&ssd_300_vgg/resblock_3_1/conv_0/biasesB'ssd_300_vgg/resblock_3_1/conv_0/weightsB&ssd_300_vgg/resblock_3_1/conv_1/biasesB'ssd_300_vgg/resblock_3_1/conv_1/weights*
dtype0
?
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?
?
save_1/AssignAssignssd_300_vgg/batch_norm_00/betasave_1/RestoreV2*
T0*1
_class'
%#loc:@ssd_300_vgg/batch_norm_00/beta*
use_locking(*
_output_shapes
:@*
validate_shape(
?
save_1/Assign_1Assignssd_300_vgg/batch_norm_00/gammasave_1/RestoreV2:1*
T0*2
_class(
&$loc:@ssd_300_vgg/batch_norm_00/gamma*
_output_shapes
:@*
use_locking(*
validate_shape(
?
save_1/Assign_2Assign%ssd_300_vgg/batch_norm_00/moving_meansave_1/RestoreV2:2*8
_class.
,*loc:@ssd_300_vgg/batch_norm_00/moving_mean*
T0*
validate_shape(*
use_locking(*
_output_shapes
:@
?
save_1/Assign_3Assign)ssd_300_vgg/batch_norm_00/moving_variancesave_1/RestoreV2:3*<
_class2
0.loc:@ssd_300_vgg/batch_norm_00/moving_variance*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0
?
save_1/Assign_4Assign'ssd_300_vgg/block10_box/conv_cls/biasessave_1/RestoreV2:4*
use_locking(*
validate_shape(*
T0*
_output_shapes
:T*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_cls/biases
?
save_1/Assign_5Assign(ssd_300_vgg/block10_box/conv_cls/weightssave_1/RestoreV2:5*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_cls/weights*'
_output_shapes
:?T*
validate_shape(*
use_locking(*
T0
?
save_1/Assign_6Assign'ssd_300_vgg/block10_box/conv_loc/biasessave_1/RestoreV2:6*:
_class0
.,loc:@ssd_300_vgg/block10_box/conv_loc/biases*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
?
save_1/Assign_7Assign(ssd_300_vgg/block10_box/conv_loc/weightssave_1/RestoreV2:7*;
_class1
/-loc:@ssd_300_vgg/block10_box/conv_loc/weights*
T0*'
_output_shapes
:?*
validate_shape(*
use_locking(
?
save_1/Assign_8Assign'ssd_300_vgg/block11_box/conv_cls/biasessave_1/RestoreV2:8*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_cls/biases*
use_locking(*
T0*
validate_shape(*
_output_shapes
:T
?
save_1/Assign_9Assign(ssd_300_vgg/block11_box/conv_cls/weightssave_1/RestoreV2:9*
validate_shape(*
T0*
use_locking(*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_cls/weights*'
_output_shapes
:?T
?
save_1/Assign_10Assign'ssd_300_vgg/block11_box/conv_loc/biasessave_1/RestoreV2:10*
validate_shape(*
use_locking(*
T0*:
_class0
.,loc:@ssd_300_vgg/block11_box/conv_loc/biases*
_output_shapes
:
?
save_1/Assign_11Assign(ssd_300_vgg/block11_box/conv_loc/weightssave_1/RestoreV2:11*'
_output_shapes
:?*
validate_shape(*
use_locking(*;
_class1
/-loc:@ssd_300_vgg/block11_box/conv_loc/weights*
T0
?
save_1/Assign_12Assign,ssd_300_vgg/block4_box/L2Normalization/gammasave_1/RestoreV2:12*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(*?
_class5
31loc:@ssd_300_vgg/block4_box/L2Normalization/gamma
?
save_1/Assign_13Assign&ssd_300_vgg/block4_box/conv_cls/biasessave_1/RestoreV2:13*
T0*
_output_shapes
:T*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_cls/biases*
validate_shape(
?
save_1/Assign_14Assign'ssd_300_vgg/block4_box/conv_cls/weightssave_1/RestoreV2:14*
T0*'
_output_shapes
:?T*
validate_shape(*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_cls/weights
?
save_1/Assign_15Assign&ssd_300_vgg/block4_box/conv_loc/biasessave_1/RestoreV2:15*9
_class/
-+loc:@ssd_300_vgg/block4_box/conv_loc/biases*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
?
save_1/Assign_16Assign'ssd_300_vgg/block4_box/conv_loc/weightssave_1/RestoreV2:16*'
_output_shapes
:?*:
_class0
.,loc:@ssd_300_vgg/block4_box/conv_loc/weights*
T0*
validate_shape(*
use_locking(
?
save_1/Assign_17Assign&ssd_300_vgg/block7_box/conv_cls/biasessave_1/RestoreV2:17*9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_cls/biases*
T0*
use_locking(*
_output_shapes
:~*
validate_shape(
?
save_1/Assign_18Assign'ssd_300_vgg/block7_box/conv_cls/weightssave_1/RestoreV2:18*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_cls/weights*
T0*'
_output_shapes
:?~*
validate_shape(
?
save_1/Assign_19Assign&ssd_300_vgg/block7_box/conv_loc/biasessave_1/RestoreV2:19*9
_class/
-+loc:@ssd_300_vgg/block7_box/conv_loc/biases*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
?
save_1/Assign_20Assign'ssd_300_vgg/block7_box/conv_loc/weightssave_1/RestoreV2:20*
T0*:
_class0
.,loc:@ssd_300_vgg/block7_box/conv_loc/weights*'
_output_shapes
:?*
use_locking(*
validate_shape(
?
save_1/Assign_21Assign&ssd_300_vgg/block8_box/conv_cls/biasessave_1/RestoreV2:21*
_output_shapes
:~*
T0*
validate_shape(*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_cls/biases
?
save_1/Assign_22Assign'ssd_300_vgg/block8_box/conv_cls/weightssave_1/RestoreV2:22*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_cls/weights*
use_locking(*'
_output_shapes
:?~*
validate_shape(*
T0
?
save_1/Assign_23Assign&ssd_300_vgg/block8_box/conv_loc/biasessave_1/RestoreV2:23*
use_locking(*
T0*9
_class/
-+loc:@ssd_300_vgg/block8_box/conv_loc/biases*
_output_shapes
:*
validate_shape(
?
save_1/Assign_24Assign'ssd_300_vgg/block8_box/conv_loc/weightssave_1/RestoreV2:24*
T0*'
_output_shapes
:?*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/block8_box/conv_loc/weights*
validate_shape(
?
save_1/Assign_25Assign&ssd_300_vgg/block9_box/conv_cls/biasessave_1/RestoreV2:25*
_output_shapes
:~*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_cls/biases*
T0*
validate_shape(*
use_locking(
?
save_1/Assign_26Assign'ssd_300_vgg/block9_box/conv_cls/weightssave_1/RestoreV2:26*
use_locking(*
validate_shape(*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_cls/weights*'
_output_shapes
:?~*
T0
?
save_1/Assign_27Assign&ssd_300_vgg/block9_box/conv_loc/biasessave_1/RestoreV2:27*9
_class/
-+loc:@ssd_300_vgg/block9_box/conv_loc/biases*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
?
save_1/Assign_28Assign'ssd_300_vgg/block9_box/conv_loc/weightssave_1/RestoreV2:28*:
_class0
.,loc:@ssd_300_vgg/block9_box/conv_loc/weights*
validate_shape(*
T0*'
_output_shapes
:?*
use_locking(
?
save_1/Assign_29Assignssd_300_vgg/conv10_1/biasessave_1/RestoreV2:29*
validate_shape(*.
_class$
" loc:@ssd_300_vgg/conv10_1/biases*
T0*
use_locking(*
_output_shapes	
:?
?
save_1/Assign_30Assignssd_300_vgg/conv10_1/weightssave_1/RestoreV2:30*(
_output_shapes
:??*
use_locking(*
T0*/
_class%
#!loc:@ssd_300_vgg/conv10_1/weights*
validate_shape(
?
save_1/Assign_31Assignssd_300_vgg/conv10_2/biasessave_1/RestoreV2:31*
_output_shapes	
:?*.
_class$
" loc:@ssd_300_vgg/conv10_2/biases*
validate_shape(*
T0*
use_locking(
?
save_1/Assign_32Assignssd_300_vgg/conv10_2/weightssave_1/RestoreV2:32*
validate_shape(*
T0*/
_class%
#!loc:@ssd_300_vgg/conv10_2/weights*(
_output_shapes
:??*
use_locking(
?
save_1/Assign_33Assignssd_300_vgg/conv8_1/biasessave_1/RestoreV2:33*-
_class#
!loc:@ssd_300_vgg/conv8_1/biases*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_34Assignssd_300_vgg/conv8_1/weightssave_1/RestoreV2:34*
validate_shape(*
T0*.
_class$
" loc:@ssd_300_vgg/conv8_1/weights*(
_output_shapes
:??*
use_locking(
?
save_1/Assign_35Assignssd_300_vgg/conv8_2/biasessave_1/RestoreV2:35*
T0*-
_class#
!loc:@ssd_300_vgg/conv8_2/biases*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_36Assignssd_300_vgg/conv8_2/weightssave_1/RestoreV2:36*.
_class$
" loc:@ssd_300_vgg/conv8_2/weights*
use_locking(*
T0*(
_output_shapes
:??*
validate_shape(
?
save_1/Assign_37Assignssd_300_vgg/conv9_1/biasessave_1/RestoreV2:37*
use_locking(*
T0*
validate_shape(*-
_class#
!loc:@ssd_300_vgg/conv9_1/biases*
_output_shapes	
:?
?
save_1/Assign_38Assignssd_300_vgg/conv9_1/weightssave_1/RestoreV2:38*.
_class$
" loc:@ssd_300_vgg/conv9_1/weights*
use_locking(*
T0*(
_output_shapes
:??*
validate_shape(
?
save_1/Assign_39Assignssd_300_vgg/conv9_2/biasessave_1/RestoreV2:39*-
_class#
!loc:@ssd_300_vgg/conv9_2/biases*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
save_1/Assign_40Assignssd_300_vgg/conv9_2/weightssave_1/RestoreV2:40*
validate_shape(*
use_locking(*(
_output_shapes
:??*.
_class$
" loc:@ssd_300_vgg/conv9_2/weights*
T0
?
save_1/Assign_41Assignssd_300_vgg/conv_init/biasessave_1/RestoreV2:41*/
_class%
#!loc:@ssd_300_vgg/conv_init/biases*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(
?
save_1/Assign_42Assignssd_300_vgg/conv_init/weightssave_1/RestoreV2:42*
validate_shape(*
T0*0
_class&
$"loc:@ssd_300_vgg/conv_init/weights*&
_output_shapes
:@*
use_locking(
?
save_1/Assign_43Assignssd_300_vgg/conv_init_1/biasessave_1/RestoreV2:43*1
_class'
%#loc:@ssd_300_vgg/conv_init_1/biases*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
?
save_1/Assign_44Assignssd_300_vgg/conv_init_1/weightssave_1/RestoreV2:44*2
_class(
&$loc:@ssd_300_vgg/conv_init_1/weights*&
_output_shapes
:@@*
use_locking(*
T0*
validate_shape(
?
save_1/Assign_45Assign)ssd_300_vgg/resblock0_0/batch_norm_0/betasave_1/RestoreV2:45*
_output_shapes
:@*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_0/beta*
T0*
validate_shape(
?
save_1/Assign_46Assign*ssd_300_vgg/resblock0_0/batch_norm_0/gammasave_1/RestoreV2:46*
T0*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_0/gamma*
use_locking(*
validate_shape(*
_output_shapes
:@
?
save_1/Assign_47Assign0ssd_300_vgg/resblock0_0/batch_norm_0/moving_meansave_1/RestoreV2:47*
T0*
_output_shapes
:@*
validate_shape(*C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean*
use_locking(
?
save_1/Assign_48Assign4ssd_300_vgg/resblock0_0/batch_norm_0/moving_variancesave_1/RestoreV2:48*
validate_shape(*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance*
use_locking(*
_output_shapes
:@*
T0
?
save_1/Assign_49Assign)ssd_300_vgg/resblock0_0/batch_norm_1/betasave_1/RestoreV2:49*
T0*
_output_shapes
:@*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock0_0/batch_norm_1/beta*
validate_shape(
?
save_1/Assign_50Assign*ssd_300_vgg/resblock0_0/batch_norm_1/gammasave_1/RestoreV2:50*=
_class3
1/loc:@ssd_300_vgg/resblock0_0/batch_norm_1/gamma*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(
?
save_1/Assign_51Assign0ssd_300_vgg/resblock0_0/batch_norm_1/moving_meansave_1/RestoreV2:51*
T0*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean*
validate_shape(*
use_locking(
?
save_1/Assign_52Assign4ssd_300_vgg/resblock0_0/batch_norm_1/moving_variancesave_1/RestoreV2:52*
T0*
validate_shape(*G
_class=
;9loc:@ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance*
_output_shapes
:@*
use_locking(
?
save_1/Assign_53Assign%ssd_300_vgg/resblock0_0/conv_0/biasessave_1/RestoreV2:53*
validate_shape(*
use_locking(*
_output_shapes
:@*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_0/biases
?
save_1/Assign_54Assign&ssd_300_vgg/resblock0_0/conv_0/weightssave_1/RestoreV2:54*
T0*
use_locking(*&
_output_shapes
:@@*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_0/weights
?
save_1/Assign_55Assign%ssd_300_vgg/resblock0_0/conv_1/biasessave_1/RestoreV2:55*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock0_0/conv_1/biases
?
save_1/Assign_56Assign&ssd_300_vgg/resblock0_0/conv_1/weightssave_1/RestoreV2:56*&
_output_shapes
:@@*9
_class/
-+loc:@ssd_300_vgg/resblock0_0/conv_1/weights*
use_locking(*
validate_shape(*
T0
?
save_1/Assign_57Assign)ssd_300_vgg/resblock0_1/batch_norm_0/betasave_1/RestoreV2:57*
_output_shapes
:@*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_0/beta*
T0*
validate_shape(*
use_locking(
?
save_1/Assign_58Assign*ssd_300_vgg/resblock0_1/batch_norm_0/gammasave_1/RestoreV2:58*
T0*
use_locking(*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_0/gamma*
_output_shapes
:@*
validate_shape(
?
save_1/Assign_59Assign0ssd_300_vgg/resblock0_1/batch_norm_0/moving_meansave_1/RestoreV2:59*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean
?
save_1/Assign_60Assign4ssd_300_vgg/resblock0_1/batch_norm_0/moving_variancesave_1/RestoreV2:60*
use_locking(*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance*
_output_shapes
:@*
validate_shape(*
T0
?
save_1/Assign_61Assign)ssd_300_vgg/resblock0_1/batch_norm_1/betasave_1/RestoreV2:61*
_output_shapes
:@*<
_class2
0.loc:@ssd_300_vgg/resblock0_1/batch_norm_1/beta*
T0*
validate_shape(*
use_locking(
?
save_1/Assign_62Assign*ssd_300_vgg/resblock0_1/batch_norm_1/gammasave_1/RestoreV2:62*
_output_shapes
:@*
T0*
use_locking(*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock0_1/batch_norm_1/gamma
?
save_1/Assign_63Assign0ssd_300_vgg/resblock0_1/batch_norm_1/moving_meansave_1/RestoreV2:63*C
_class9
75loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
?
save_1/Assign_64Assign4ssd_300_vgg/resblock0_1/batch_norm_1/moving_variancesave_1/RestoreV2:64*
T0*
validate_shape(*G
_class=
;9loc:@ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance*
_output_shapes
:@*
use_locking(
?
save_1/Assign_65Assign%ssd_300_vgg/resblock0_1/conv_0/biasessave_1/RestoreV2:65*
use_locking(*
T0*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_0/biases*
_output_shapes
:@
?
save_1/Assign_66Assign&ssd_300_vgg/resblock0_1/conv_0/weightssave_1/RestoreV2:66*
use_locking(*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_0/weights*&
_output_shapes
:@@*
validate_shape(
?
save_1/Assign_67Assign%ssd_300_vgg/resblock0_1/conv_1/biasessave_1/RestoreV2:67*
use_locking(*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock0_1/conv_1/biases*
validate_shape(*
_output_shapes
:@
?
save_1/Assign_68Assign&ssd_300_vgg/resblock0_1/conv_1/weightssave_1/RestoreV2:68*
use_locking(*&
_output_shapes
:@@*9
_class/
-+loc:@ssd_300_vgg/resblock0_1/conv_1/weights*
validate_shape(*
T0
?
save_1/Assign_69Assign)ssd_300_vgg/resblock1_0/batch_norm_0/betasave_1/RestoreV2:69*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_0/beta*
_output_shapes
:@*
validate_shape(*
T0*
use_locking(
?
save_1/Assign_70Assign*ssd_300_vgg/resblock1_0/batch_norm_0/gammasave_1/RestoreV2:70*
validate_shape(*
_output_shapes
:@*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_0/gamma*
use_locking(*
T0
?
save_1/Assign_71Assign0ssd_300_vgg/resblock1_0/batch_norm_0/moving_meansave_1/RestoreV2:71*
use_locking(*
T0*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean*
validate_shape(*
_output_shapes
:@
?
save_1/Assign_72Assign4ssd_300_vgg/resblock1_0/batch_norm_0/moving_variancesave_1/RestoreV2:72*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance*
use_locking(*
_output_shapes
:@*
validate_shape(*
T0
?
save_1/Assign_73Assign)ssd_300_vgg/resblock1_0/batch_norm_1/betasave_1/RestoreV2:73*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/batch_norm_1/beta*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
save_1/Assign_74Assign*ssd_300_vgg/resblock1_0/batch_norm_1/gammasave_1/RestoreV2:74*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock1_0/batch_norm_1/gamma*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_75Assign0ssd_300_vgg/resblock1_0/batch_norm_1/moving_meansave_1/RestoreV2:75*
T0*C
_class9
75loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean*
validate_shape(*
use_locking(*
_output_shapes	
:?
?
save_1/Assign_76Assign4ssd_300_vgg/resblock1_0/batch_norm_1/moving_variancesave_1/RestoreV2:76*
use_locking(*
T0*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance*
validate_shape(
?
save_1/Assign_77Assign%ssd_300_vgg/resblock1_0/conv_0/biasessave_1/RestoreV2:77*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_0/biases*
T0*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_78Assign&ssd_300_vgg/resblock1_0/conv_0/weightssave_1/RestoreV2:78*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_0/weights*'
_output_shapes
:@?*
use_locking(*
validate_shape(
?
save_1/Assign_79Assign%ssd_300_vgg/resblock1_0/conv_1/biasessave_1/RestoreV2:79*
_output_shapes	
:?*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock1_0/conv_1/biases*
T0*
use_locking(
?
save_1/Assign_80Assign&ssd_300_vgg/resblock1_0/conv_1/weightssave_1/RestoreV2:80*
use_locking(*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock1_0/conv_1/weights*
validate_shape(*(
_output_shapes
:??
?
save_1/Assign_81Assign(ssd_300_vgg/resblock1_0/conv_init/biasessave_1/RestoreV2:81*
validate_shape(*;
_class1
/-loc:@ssd_300_vgg/resblock1_0/conv_init/biases*
_output_shapes	
:?*
use_locking(*
T0
?
save_1/Assign_82Assign)ssd_300_vgg/resblock1_0/conv_init/weightssave_1/RestoreV2:82*
validate_shape(*'
_output_shapes
:@?*<
_class2
0.loc:@ssd_300_vgg/resblock1_0/conv_init/weights*
T0*
use_locking(
?
save_1/Assign_83Assign)ssd_300_vgg/resblock1_1/batch_norm_0/betasave_1/RestoreV2:83*
T0*
_output_shapes	
:?*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_0/beta*
use_locking(*
validate_shape(
?
save_1/Assign_84Assign*ssd_300_vgg/resblock1_1/batch_norm_0/gammasave_1/RestoreV2:84*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_0/gamma*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0
?
save_1/Assign_85Assign0ssd_300_vgg/resblock1_1/batch_norm_0/moving_meansave_1/RestoreV2:85*
T0*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_86Assign4ssd_300_vgg/resblock1_1/batch_norm_0/moving_variancesave_1/RestoreV2:86*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
save_1/Assign_87Assign)ssd_300_vgg/resblock1_1/batch_norm_1/betasave_1/RestoreV2:87*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock1_1/batch_norm_1/beta
?
save_1/Assign_88Assign*ssd_300_vgg/resblock1_1/batch_norm_1/gammasave_1/RestoreV2:88*=
_class3
1/loc:@ssd_300_vgg/resblock1_1/batch_norm_1/gamma*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
save_1/Assign_89Assign0ssd_300_vgg/resblock1_1/batch_norm_1/moving_meansave_1/RestoreV2:89*
use_locking(*
T0*C
_class9
75loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_90Assign4ssd_300_vgg/resblock1_1/batch_norm_1/moving_variancesave_1/RestoreV2:90*
validate_shape(*
use_locking(*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance*
_output_shapes	
:?
?
save_1/Assign_91Assign%ssd_300_vgg/resblock1_1/conv_0/biasessave_1/RestoreV2:91*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_0/biases*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0
?
save_1/Assign_92Assign&ssd_300_vgg/resblock1_1/conv_0/weightssave_1/RestoreV2:92*
use_locking(*
validate_shape(*(
_output_shapes
:??*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_0/weights
?
save_1/Assign_93Assign%ssd_300_vgg/resblock1_1/conv_1/biasessave_1/RestoreV2:93*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock1_1/conv_1/biases
?
save_1/Assign_94Assign&ssd_300_vgg/resblock1_1/conv_1/weightssave_1/RestoreV2:94*
validate_shape(*
T0*(
_output_shapes
:??*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock1_1/conv_1/weights
?
save_1/Assign_95Assign)ssd_300_vgg/resblock2_0/batch_norm_0/betasave_1/RestoreV2:95*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_0/beta*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(
?
save_1/Assign_96Assign*ssd_300_vgg/resblock2_0/batch_norm_0/gammasave_1/RestoreV2:96*
use_locking(*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_0/gamma*
T0*
_output_shapes	
:?
?
save_1/Assign_97Assign0ssd_300_vgg/resblock2_0/batch_norm_0/moving_meansave_1/RestoreV2:97*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean*
validate_shape(*
use_locking(*
T0
?
save_1/Assign_98Assign4ssd_300_vgg/resblock2_0/batch_norm_0/moving_variancesave_1/RestoreV2:98*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance
?
save_1/Assign_99Assign)ssd_300_vgg/resblock2_0/batch_norm_1/betasave_1/RestoreV2:99*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/batch_norm_1/beta
?
save_1/Assign_100Assign*ssd_300_vgg/resblock2_0/batch_norm_1/gammasave_1/RestoreV2:100*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock2_0/batch_norm_1/gamma*
T0*
use_locking(*
_output_shapes	
:?
?
save_1/Assign_101Assign0ssd_300_vgg/resblock2_0/batch_norm_1/moving_meansave_1/RestoreV2:101*
_output_shapes	
:?*
validate_shape(*C
_class9
75loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean*
T0*
use_locking(
?
save_1/Assign_102Assign4ssd_300_vgg/resblock2_0/batch_norm_1/moving_variancesave_1/RestoreV2:102*
validate_shape(*
T0*
use_locking(*G
_class=
;9loc:@ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance*
_output_shapes	
:?
?
save_1/Assign_103Assign%ssd_300_vgg/resblock2_0/conv_0/biasessave_1/RestoreV2:103*
T0*
use_locking(*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_0/biases*
_output_shapes	
:?
?
save_1/Assign_104Assign&ssd_300_vgg/resblock2_0/conv_0/weightssave_1/RestoreV2:104*
use_locking(*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_0/weights*(
_output_shapes
:??*
T0
?
save_1/Assign_105Assign%ssd_300_vgg/resblock2_0/conv_1/biasessave_1/RestoreV2:105*
use_locking(*
_output_shapes	
:?*
validate_shape(*8
_class.
,*loc:@ssd_300_vgg/resblock2_0/conv_1/biases*
T0
?
save_1/Assign_106Assign&ssd_300_vgg/resblock2_0/conv_1/weightssave_1/RestoreV2:106*9
_class/
-+loc:@ssd_300_vgg/resblock2_0/conv_1/weights*(
_output_shapes
:??*
validate_shape(*
use_locking(*
T0
?
save_1/Assign_107Assign(ssd_300_vgg/resblock2_0/conv_init/biasessave_1/RestoreV2:107*
use_locking(*
T0*
validate_shape(*;
_class1
/-loc:@ssd_300_vgg/resblock2_0/conv_init/biases*
_output_shapes	
:?
?
save_1/Assign_108Assign)ssd_300_vgg/resblock2_0/conv_init/weightssave_1/RestoreV2:108*
use_locking(*<
_class2
0.loc:@ssd_300_vgg/resblock2_0/conv_init/weights*
validate_shape(*(
_output_shapes
:??*
T0
?
save_1/Assign_109Assign)ssd_300_vgg/resblock2_1/batch_norm_0/betasave_1/RestoreV2:109*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_0/beta*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_110Assign*ssd_300_vgg/resblock2_1/batch_norm_0/gammasave_1/RestoreV2:110*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_0/gamma
?
save_1/Assign_111Assign0ssd_300_vgg/resblock2_1/batch_norm_0/moving_meansave_1/RestoreV2:111*
_output_shapes	
:?*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean*
use_locking(*
validate_shape(*
T0
?
save_1/Assign_112Assign4ssd_300_vgg/resblock2_1/batch_norm_0/moving_variancesave_1/RestoreV2:112*
T0*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_113Assign)ssd_300_vgg/resblock2_1/batch_norm_1/betasave_1/RestoreV2:113*
validate_shape(*
use_locking(*
T0*<
_class2
0.loc:@ssd_300_vgg/resblock2_1/batch_norm_1/beta*
_output_shapes	
:?
?
save_1/Assign_114Assign*ssd_300_vgg/resblock2_1/batch_norm_1/gammasave_1/RestoreV2:114*
use_locking(*=
_class3
1/loc:@ssd_300_vgg/resblock2_1/batch_norm_1/gamma*
T0*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_115Assign0ssd_300_vgg/resblock2_1/batch_norm_1/moving_meansave_1/RestoreV2:115*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0*C
_class9
75loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean
?
save_1/Assign_116Assign4ssd_300_vgg/resblock2_1/batch_norm_1/moving_variancesave_1/RestoreV2:116*
_output_shapes	
:?*
validate_shape(*
T0*
use_locking(*G
_class=
;9loc:@ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance
?
save_1/Assign_117Assign%ssd_300_vgg/resblock2_1/conv_0/biasessave_1/RestoreV2:117*
validate_shape(*
T0*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_0/biases*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_118Assign&ssd_300_vgg/resblock2_1/conv_0/weightssave_1/RestoreV2:118*(
_output_shapes
:??*
validate_shape(*
T0*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_0/weights
?
save_1/Assign_119Assign%ssd_300_vgg/resblock2_1/conv_1/biasessave_1/RestoreV2:119*
_output_shapes	
:?*8
_class.
,*loc:@ssd_300_vgg/resblock2_1/conv_1/biases*
T0*
use_locking(*
validate_shape(
?
save_1/Assign_120Assign&ssd_300_vgg/resblock2_1/conv_1/weightssave_1/RestoreV2:120*
T0*(
_output_shapes
:??*
validate_shape(*9
_class/
-+loc:@ssd_300_vgg/resblock2_1/conv_1/weights*
use_locking(
?
save_1/Assign_121Assign*ssd_300_vgg/resblock_3_0/batch_norm_0/betasave_1/RestoreV2:121*
_output_shapes	
:?*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/beta*
T0*
validate_shape(*
use_locking(
?
save_1/Assign_122Assign+ssd_300_vgg/resblock_3_0/batch_norm_0/gammasave_1/RestoreV2:122*
validate_shape(*
T0*
use_locking(*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/gamma*
_output_shapes	
:?
?
save_1/Assign_123Assign1ssd_300_vgg/resblock_3_0/batch_norm_0/moving_meansave_1/RestoreV2:123*
T0*
use_locking(*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_124Assign5ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variancesave_1/RestoreV2:124*
use_locking(*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance*
_output_shapes	
:?*
validate_shape(*
T0
?
save_1/Assign_125Assign*ssd_300_vgg/resblock_3_0/batch_norm_1/betasave_1/RestoreV2:125*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/beta*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_126Assign+ssd_300_vgg/resblock_3_0/batch_norm_1/gammasave_1/RestoreV2:126*
validate_shape(*>
_class4
20loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/gamma*
T0*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_127Assign1ssd_300_vgg/resblock_3_0/batch_norm_1/moving_meansave_1/RestoreV2:127*
T0*D
_class:
86loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_128Assign5ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variancesave_1/RestoreV2:128*H
_class>
<:loc:@ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(
?
save_1/Assign_129Assign&ssd_300_vgg/resblock_3_0/conv_0/biasessave_1/RestoreV2:129*
validate_shape(*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_0/biases*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_130Assign'ssd_300_vgg/resblock_3_0/conv_0/weightssave_1/RestoreV2:130*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_0/weights*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0
?
save_1/Assign_131Assign&ssd_300_vgg/resblock_3_0/conv_1/biasessave_1/RestoreV2:131*
_output_shapes	
:?*
T0*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock_3_0/conv_1/biases*
validate_shape(
?
save_1/Assign_132Assign'ssd_300_vgg/resblock_3_0/conv_1/weightssave_1/RestoreV2:132*
T0*:
_class0
.,loc:@ssd_300_vgg/resblock_3_0/conv_1/weights*(
_output_shapes
:??*
use_locking(*
validate_shape(
?
save_1/Assign_133Assign)ssd_300_vgg/resblock_3_0/conv_init/biasessave_1/RestoreV2:133*<
_class2
0.loc:@ssd_300_vgg/resblock_3_0/conv_init/biases*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(
?
save_1/Assign_134Assign*ssd_300_vgg/resblock_3_0/conv_init/weightssave_1/RestoreV2:134*
use_locking(*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock_3_0/conv_init/weights*
T0*(
_output_shapes
:??
?
save_1/Assign_135Assign*ssd_300_vgg/resblock_3_1/batch_norm_0/betasave_1/RestoreV2:135*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/beta
?
save_1/Assign_136Assign+ssd_300_vgg/resblock_3_1/batch_norm_0/gammasave_1/RestoreV2:136*
T0*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/gamma*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_137Assign1ssd_300_vgg/resblock_3_1/batch_norm_0/moving_meansave_1/RestoreV2:137*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean
?
save_1/Assign_138Assign5ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variancesave_1/RestoreV2:138*
T0*
validate_shape(*
use_locking(*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance*
_output_shapes	
:?
?
save_1/Assign_139Assign*ssd_300_vgg/resblock_3_1/batch_norm_1/betasave_1/RestoreV2:139*=
_class3
1/loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/beta*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_140Assign+ssd_300_vgg/resblock_3_1/batch_norm_1/gammasave_1/RestoreV2:140*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?*>
_class4
20loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/gamma
?
save_1/Assign_141Assign1ssd_300_vgg/resblock_3_1/batch_norm_1/moving_meansave_1/RestoreV2:141*
T0*
use_locking(*D
_class:
86loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_142Assign5ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variancesave_1/RestoreV2:142*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(*H
_class>
<:loc:@ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance
?
save_1/Assign_143Assign&ssd_300_vgg/resblock_3_1/conv_0/biasessave_1/RestoreV2:143*
use_locking(*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_0/biases*
_output_shapes	
:?*
validate_shape(*
T0
?
save_1/Assign_144Assign'ssd_300_vgg/resblock_3_1/conv_0/weightssave_1/RestoreV2:144*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_0/weights*
use_locking(*
validate_shape(*
T0*(
_output_shapes
:??
?
save_1/Assign_145Assign&ssd_300_vgg/resblock_3_1/conv_1/biasessave_1/RestoreV2:145*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@ssd_300_vgg/resblock_3_1/conv_1/biases*
validate_shape(
?
save_1/Assign_146Assign'ssd_300_vgg/resblock_3_1/conv_1/weightssave_1/RestoreV2:146*(
_output_shapes
:??*
T0*
use_locking(*:
_class0
.,loc:@ssd_300_vgg/resblock_3_1/conv_1/weights*
validate_shape(
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_100^save_1/Assign_101^save_1/Assign_102^save_1/Assign_103^save_1/Assign_104^save_1/Assign_105^save_1/Assign_106^save_1/Assign_107^save_1/Assign_108^save_1/Assign_109^save_1/Assign_11^save_1/Assign_110^save_1/Assign_111^save_1/Assign_112^save_1/Assign_113^save_1/Assign_114^save_1/Assign_115^save_1/Assign_116^save_1/Assign_117^save_1/Assign_118^save_1/Assign_119^save_1/Assign_12^save_1/Assign_120^save_1/Assign_121^save_1/Assign_122^save_1/Assign_123^save_1/Assign_124^save_1/Assign_125^save_1/Assign_126^save_1/Assign_127^save_1/Assign_128^save_1/Assign_129^save_1/Assign_13^save_1/Assign_130^save_1/Assign_131^save_1/Assign_132^save_1/Assign_133^save_1/Assign_134^save_1/Assign_135^save_1/Assign_136^save_1/Assign_137^save_1/Assign_138^save_1/Assign_139^save_1/Assign_14^save_1/Assign_140^save_1/Assign_141^save_1/Assign_142^save_1/Assign_143^save_1/Assign_144^save_1/Assign_145^save_1/Assign_146^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_8^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_84^save_1/Assign_85^save_1/Assign_86^save_1/Assign_87^save_1/Assign_88^save_1/Assign_89^save_1/Assign_9^save_1/Assign_90^save_1/Assign_91^save_1/Assign_92^save_1/Assign_93^save_1/Assign_94^save_1/Assign_95^save_1/Assign_96^save_1/Assign_97^save_1/Assign_98^save_1/Assign_99
1
save_1/restore_allNoOp^save_1/restore_shard "B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"?
regularization_losses?
?
9ssd_300_vgg/conv_init/kernel/Regularizer/l2_regularizer:0
;ssd_300_vgg/conv_init_1/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock0_0/conv_0/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock0_0/conv_1/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock0_1/conv_0/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock0_1/conv_1/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock1_0/conv_0/kernel/Regularizer/l2_regularizer:0
Essd_300_vgg/resblock1_0/conv_init/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock1_0/conv_1/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock1_1/conv_0/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock1_1/conv_1/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock2_0/conv_0/kernel/Regularizer/l2_regularizer:0
Essd_300_vgg/resblock2_0/conv_init/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock2_0/conv_1/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock2_1/conv_0/kernel/Regularizer/l2_regularizer:0
Bssd_300_vgg/resblock2_1/conv_1/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/resblock_3_0/conv_0/kernel/Regularizer/l2_regularizer:0
Fssd_300_vgg/resblock_3_0/conv_init/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/resblock_3_0/conv_1/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/resblock_3_1/conv_0/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/resblock_3_1/conv_1/kernel/Regularizer/l2_regularizer:0
7ssd_300_vgg/conv8_1/kernel/Regularizer/l2_regularizer:0
7ssd_300_vgg/conv8_2/kernel/Regularizer/l2_regularizer:0
7ssd_300_vgg/conv9_1/kernel/Regularizer/l2_regularizer:0
7ssd_300_vgg/conv9_2/kernel/Regularizer/l2_regularizer:0
8ssd_300_vgg/conv10_1/kernel/Regularizer/l2_regularizer:0
8ssd_300_vgg/conv10_2/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/block4_box/conv_loc/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/block4_box/conv_cls/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/block7_box/conv_loc/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/block7_box/conv_cls/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/block8_box/conv_loc/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/block8_box/conv_cls/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/block9_box/conv_loc/kernel/Regularizer/l2_regularizer:0
Cssd_300_vgg/block9_box/conv_cls/kernel/Regularizer/l2_regularizer:0
Dssd_300_vgg/block10_box/conv_loc/kernel/Regularizer/l2_regularizer:0
Dssd_300_vgg/block10_box/conv_cls/kernel/Regularizer/l2_regularizer:0
Dssd_300_vgg/block11_box/conv_loc/kernel/Regularizer/l2_regularizer:0
Dssd_300_vgg/block11_box/conv_cls/kernel/Regularizer/l2_regularizer:0"??
model_variables????
?
ssd_300_vgg/conv_init/weights:0$ssd_300_vgg/conv_init/weights/Assign$ssd_300_vgg/conv_init/weights/read:02:ssd_300_vgg/conv_init/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv_init/biases:0#ssd_300_vgg/conv_init/biases/Assign#ssd_300_vgg/conv_init/biases/read:020ssd_300_vgg/conv_init/biases/Initializer/zeros:08
?
 ssd_300_vgg/batch_norm_00/beta:0%ssd_300_vgg/batch_norm_00/beta/Assign%ssd_300_vgg/batch_norm_00/beta/read:022ssd_300_vgg/batch_norm_00/beta/Initializer/zeros:08
?
!ssd_300_vgg/batch_norm_00/gamma:0&ssd_300_vgg/batch_norm_00/gamma/Assign&ssd_300_vgg/batch_norm_00/gamma/read:022ssd_300_vgg/batch_norm_00/gamma/Initializer/ones:08
?
'ssd_300_vgg/batch_norm_00/moving_mean:0,ssd_300_vgg/batch_norm_00/moving_mean/Assign,ssd_300_vgg/batch_norm_00/moving_mean/read:029ssd_300_vgg/batch_norm_00/moving_mean/Initializer/zeros:0
?
+ssd_300_vgg/batch_norm_00/moving_variance:00ssd_300_vgg/batch_norm_00/moving_variance/Assign0ssd_300_vgg/batch_norm_00/moving_variance/read:02<ssd_300_vgg/batch_norm_00/moving_variance/Initializer/ones:0
?
!ssd_300_vgg/conv_init_1/weights:0&ssd_300_vgg/conv_init_1/weights/Assign&ssd_300_vgg/conv_init_1/weights/read:02<ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform:08
?
 ssd_300_vgg/conv_init_1/biases:0%ssd_300_vgg/conv_init_1/biases/Assign%ssd_300_vgg/conv_init_1/biases/read:022ssd_300_vgg/conv_init_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_0/batch_norm_0/beta:00ssd_300_vgg/resblock0_0/batch_norm_0/beta/Assign0ssd_300_vgg/resblock0_0/batch_norm_0/beta/read:02=ssd_300_vgg/resblock0_0/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_0/batch_norm_0/gamma:01ssd_300_vgg/resblock0_0/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock0_0/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock0_0/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean:07ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock0_0/conv_0/weights:0-ssd_300_vgg/resblock0_0/conv_0/weights/Assign-ssd_300_vgg/resblock0_0/conv_0/weights/read:02Cssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_0/conv_0/biases:0,ssd_300_vgg/resblock0_0/conv_0/biases/Assign,ssd_300_vgg/resblock0_0/conv_0/biases/read:029ssd_300_vgg/resblock0_0/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_0/batch_norm_1/beta:00ssd_300_vgg/resblock0_0/batch_norm_1/beta/Assign0ssd_300_vgg/resblock0_0/batch_norm_1/beta/read:02=ssd_300_vgg/resblock0_0/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_0/batch_norm_1/gamma:01ssd_300_vgg/resblock0_0/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock0_0/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock0_0/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean:07ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock0_0/conv_1/weights:0-ssd_300_vgg/resblock0_0/conv_1/weights/Assign-ssd_300_vgg/resblock0_0/conv_1/weights/read:02Cssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_0/conv_1/biases:0,ssd_300_vgg/resblock0_0/conv_1/biases/Assign,ssd_300_vgg/resblock0_0/conv_1/biases/read:029ssd_300_vgg/resblock0_0/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_1/batch_norm_0/beta:00ssd_300_vgg/resblock0_1/batch_norm_0/beta/Assign0ssd_300_vgg/resblock0_1/batch_norm_0/beta/read:02=ssd_300_vgg/resblock0_1/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_1/batch_norm_0/gamma:01ssd_300_vgg/resblock0_1/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock0_1/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock0_1/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean:07ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock0_1/conv_0/weights:0-ssd_300_vgg/resblock0_1/conv_0/weights/Assign-ssd_300_vgg/resblock0_1/conv_0/weights/read:02Cssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_1/conv_0/biases:0,ssd_300_vgg/resblock0_1/conv_0/biases/Assign,ssd_300_vgg/resblock0_1/conv_0/biases/read:029ssd_300_vgg/resblock0_1/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_1/batch_norm_1/beta:00ssd_300_vgg/resblock0_1/batch_norm_1/beta/Assign0ssd_300_vgg/resblock0_1/batch_norm_1/beta/read:02=ssd_300_vgg/resblock0_1/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_1/batch_norm_1/gamma:01ssd_300_vgg/resblock0_1/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock0_1/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock0_1/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean:07ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock0_1/conv_1/weights:0-ssd_300_vgg/resblock0_1/conv_1/weights/Assign-ssd_300_vgg/resblock0_1/conv_1/weights/read:02Cssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_1/conv_1/biases:0,ssd_300_vgg/resblock0_1/conv_1/biases/Assign,ssd_300_vgg/resblock0_1/conv_1/biases/read:029ssd_300_vgg/resblock0_1/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_0/batch_norm_0/beta:00ssd_300_vgg/resblock1_0/batch_norm_0/beta/Assign0ssd_300_vgg/resblock1_0/batch_norm_0/beta/read:02=ssd_300_vgg/resblock1_0/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_0/batch_norm_0/gamma:01ssd_300_vgg/resblock1_0/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock1_0/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock1_0/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean:07ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock1_0/conv_0/weights:0-ssd_300_vgg/resblock1_0/conv_0/weights/Assign-ssd_300_vgg/resblock1_0/conv_0/weights/read:02Cssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_0/conv_0/biases:0,ssd_300_vgg/resblock1_0/conv_0/biases/Assign,ssd_300_vgg/resblock1_0/conv_0/biases/read:029ssd_300_vgg/resblock1_0/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_0/conv_init/weights:00ssd_300_vgg/resblock1_0/conv_init/weights/Assign0ssd_300_vgg/resblock1_0/conv_init/weights/read:02Fssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform:08
?
*ssd_300_vgg/resblock1_0/conv_init/biases:0/ssd_300_vgg/resblock1_0/conv_init/biases/Assign/ssd_300_vgg/resblock1_0/conv_init/biases/read:02<ssd_300_vgg/resblock1_0/conv_init/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_0/batch_norm_1/beta:00ssd_300_vgg/resblock1_0/batch_norm_1/beta/Assign0ssd_300_vgg/resblock1_0/batch_norm_1/beta/read:02=ssd_300_vgg/resblock1_0/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_0/batch_norm_1/gamma:01ssd_300_vgg/resblock1_0/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock1_0/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock1_0/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean:07ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock1_0/conv_1/weights:0-ssd_300_vgg/resblock1_0/conv_1/weights/Assign-ssd_300_vgg/resblock1_0/conv_1/weights/read:02Cssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_0/conv_1/biases:0,ssd_300_vgg/resblock1_0/conv_1/biases/Assign,ssd_300_vgg/resblock1_0/conv_1/biases/read:029ssd_300_vgg/resblock1_0/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_1/batch_norm_0/beta:00ssd_300_vgg/resblock1_1/batch_norm_0/beta/Assign0ssd_300_vgg/resblock1_1/batch_norm_0/beta/read:02=ssd_300_vgg/resblock1_1/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_1/batch_norm_0/gamma:01ssd_300_vgg/resblock1_1/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock1_1/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock1_1/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean:07ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock1_1/conv_0/weights:0-ssd_300_vgg/resblock1_1/conv_0/weights/Assign-ssd_300_vgg/resblock1_1/conv_0/weights/read:02Cssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_1/conv_0/biases:0,ssd_300_vgg/resblock1_1/conv_0/biases/Assign,ssd_300_vgg/resblock1_1/conv_0/biases/read:029ssd_300_vgg/resblock1_1/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_1/batch_norm_1/beta:00ssd_300_vgg/resblock1_1/batch_norm_1/beta/Assign0ssd_300_vgg/resblock1_1/batch_norm_1/beta/read:02=ssd_300_vgg/resblock1_1/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_1/batch_norm_1/gamma:01ssd_300_vgg/resblock1_1/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock1_1/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock1_1/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean:07ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock1_1/conv_1/weights:0-ssd_300_vgg/resblock1_1/conv_1/weights/Assign-ssd_300_vgg/resblock1_1/conv_1/weights/read:02Cssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_1/conv_1/biases:0,ssd_300_vgg/resblock1_1/conv_1/biases/Assign,ssd_300_vgg/resblock1_1/conv_1/biases/read:029ssd_300_vgg/resblock1_1/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_0/batch_norm_0/beta:00ssd_300_vgg/resblock2_0/batch_norm_0/beta/Assign0ssd_300_vgg/resblock2_0/batch_norm_0/beta/read:02=ssd_300_vgg/resblock2_0/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_0/batch_norm_0/gamma:01ssd_300_vgg/resblock2_0/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock2_0/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock2_0/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean:07ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock2_0/conv_0/weights:0-ssd_300_vgg/resblock2_0/conv_0/weights/Assign-ssd_300_vgg/resblock2_0/conv_0/weights/read:02Cssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_0/conv_0/biases:0,ssd_300_vgg/resblock2_0/conv_0/biases/Assign,ssd_300_vgg/resblock2_0/conv_0/biases/read:029ssd_300_vgg/resblock2_0/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_0/conv_init/weights:00ssd_300_vgg/resblock2_0/conv_init/weights/Assign0ssd_300_vgg/resblock2_0/conv_init/weights/read:02Fssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform:08
?
*ssd_300_vgg/resblock2_0/conv_init/biases:0/ssd_300_vgg/resblock2_0/conv_init/biases/Assign/ssd_300_vgg/resblock2_0/conv_init/biases/read:02<ssd_300_vgg/resblock2_0/conv_init/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_0/batch_norm_1/beta:00ssd_300_vgg/resblock2_0/batch_norm_1/beta/Assign0ssd_300_vgg/resblock2_0/batch_norm_1/beta/read:02=ssd_300_vgg/resblock2_0/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_0/batch_norm_1/gamma:01ssd_300_vgg/resblock2_0/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock2_0/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock2_0/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean:07ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock2_0/conv_1/weights:0-ssd_300_vgg/resblock2_0/conv_1/weights/Assign-ssd_300_vgg/resblock2_0/conv_1/weights/read:02Cssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_0/conv_1/biases:0,ssd_300_vgg/resblock2_0/conv_1/biases/Assign,ssd_300_vgg/resblock2_0/conv_1/biases/read:029ssd_300_vgg/resblock2_0/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_1/batch_norm_0/beta:00ssd_300_vgg/resblock2_1/batch_norm_0/beta/Assign0ssd_300_vgg/resblock2_1/batch_norm_0/beta/read:02=ssd_300_vgg/resblock2_1/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_1/batch_norm_0/gamma:01ssd_300_vgg/resblock2_1/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock2_1/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock2_1/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean:07ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock2_1/conv_0/weights:0-ssd_300_vgg/resblock2_1/conv_0/weights/Assign-ssd_300_vgg/resblock2_1/conv_0/weights/read:02Cssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_1/conv_0/biases:0,ssd_300_vgg/resblock2_1/conv_0/biases/Assign,ssd_300_vgg/resblock2_1/conv_0/biases/read:029ssd_300_vgg/resblock2_1/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_1/batch_norm_1/beta:00ssd_300_vgg/resblock2_1/batch_norm_1/beta/Assign0ssd_300_vgg/resblock2_1/batch_norm_1/beta/read:02=ssd_300_vgg/resblock2_1/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_1/batch_norm_1/gamma:01ssd_300_vgg/resblock2_1/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock2_1/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock2_1/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean:07ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock2_1/conv_1/weights:0-ssd_300_vgg/resblock2_1/conv_1/weights/Assign-ssd_300_vgg/resblock2_1/conv_1/weights/read:02Cssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_1/conv_1/biases:0,ssd_300_vgg/resblock2_1/conv_1/biases/Assign,ssd_300_vgg/resblock2_1/conv_1/biases/read:029ssd_300_vgg/resblock2_1/conv_1/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_0/batch_norm_0/beta:01ssd_300_vgg/resblock_3_0/batch_norm_0/beta/Assign1ssd_300_vgg/resblock_3_0/batch_norm_0/beta/read:02>ssd_300_vgg/resblock_3_0/batch_norm_0/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_0/batch_norm_0/gamma:02ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/Assign2ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/read:02>ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/Initializer/ones:08
?
3ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean:08ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/Assign8ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/read:02Essd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/Initializer/zeros:0
?
7ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance:0<ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/Assign<ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/read:02Hssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/Initializer/ones:0
?
)ssd_300_vgg/resblock_3_0/conv_0/weights:0.ssd_300_vgg/resblock_3_0/conv_0/weights/Assign.ssd_300_vgg/resblock_3_0/conv_0/weights/read:02Dssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_0/conv_0/biases:0-ssd_300_vgg/resblock_3_0/conv_0/biases/Assign-ssd_300_vgg/resblock_3_0/conv_0/biases/read:02:ssd_300_vgg/resblock_3_0/conv_0/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_0/conv_init/weights:01ssd_300_vgg/resblock_3_0/conv_init/weights/Assign1ssd_300_vgg/resblock_3_0/conv_init/weights/read:02Gssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform:08
?
+ssd_300_vgg/resblock_3_0/conv_init/biases:00ssd_300_vgg/resblock_3_0/conv_init/biases/Assign0ssd_300_vgg/resblock_3_0/conv_init/biases/read:02=ssd_300_vgg/resblock_3_0/conv_init/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_0/batch_norm_1/beta:01ssd_300_vgg/resblock_3_0/batch_norm_1/beta/Assign1ssd_300_vgg/resblock_3_0/batch_norm_1/beta/read:02>ssd_300_vgg/resblock_3_0/batch_norm_1/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_0/batch_norm_1/gamma:02ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/Assign2ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/read:02>ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/Initializer/ones:08
?
3ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean:08ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/Assign8ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/read:02Essd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/Initializer/zeros:0
?
7ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance:0<ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/Assign<ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/read:02Hssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/Initializer/ones:0
?
)ssd_300_vgg/resblock_3_0/conv_1/weights:0.ssd_300_vgg/resblock_3_0/conv_1/weights/Assign.ssd_300_vgg/resblock_3_0/conv_1/weights/read:02Dssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_0/conv_1/biases:0-ssd_300_vgg/resblock_3_0/conv_1/biases/Assign-ssd_300_vgg/resblock_3_0/conv_1/biases/read:02:ssd_300_vgg/resblock_3_0/conv_1/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_1/batch_norm_0/beta:01ssd_300_vgg/resblock_3_1/batch_norm_0/beta/Assign1ssd_300_vgg/resblock_3_1/batch_norm_0/beta/read:02>ssd_300_vgg/resblock_3_1/batch_norm_0/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_1/batch_norm_0/gamma:02ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/Assign2ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/read:02>ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/Initializer/ones:08
?
3ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean:08ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/Assign8ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/read:02Essd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/Initializer/zeros:0
?
7ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance:0<ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/Assign<ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/read:02Hssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/Initializer/ones:0
?
)ssd_300_vgg/resblock_3_1/conv_0/weights:0.ssd_300_vgg/resblock_3_1/conv_0/weights/Assign.ssd_300_vgg/resblock_3_1/conv_0/weights/read:02Dssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_1/conv_0/biases:0-ssd_300_vgg/resblock_3_1/conv_0/biases/Assign-ssd_300_vgg/resblock_3_1/conv_0/biases/read:02:ssd_300_vgg/resblock_3_1/conv_0/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_1/batch_norm_1/beta:01ssd_300_vgg/resblock_3_1/batch_norm_1/beta/Assign1ssd_300_vgg/resblock_3_1/batch_norm_1/beta/read:02>ssd_300_vgg/resblock_3_1/batch_norm_1/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_1/batch_norm_1/gamma:02ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/Assign2ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/read:02>ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/Initializer/ones:08
?
3ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean:08ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/Assign8ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/read:02Essd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/Initializer/zeros:0
?
7ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance:0<ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/Assign<ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/read:02Hssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/Initializer/ones:0
?
)ssd_300_vgg/resblock_3_1/conv_1/weights:0.ssd_300_vgg/resblock_3_1/conv_1/weights/Assign.ssd_300_vgg/resblock_3_1/conv_1/weights/read:02Dssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_1/conv_1/biases:0-ssd_300_vgg/resblock_3_1/conv_1/biases/Assign-ssd_300_vgg/resblock_3_1/conv_1/biases/read:02:ssd_300_vgg/resblock_3_1/conv_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv8_1/weights:0"ssd_300_vgg/conv8_1/weights/Assign"ssd_300_vgg/conv8_1/weights/read:028ssd_300_vgg/conv8_1/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv8_1/biases:0!ssd_300_vgg/conv8_1/biases/Assign!ssd_300_vgg/conv8_1/biases/read:02.ssd_300_vgg/conv8_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv8_2/weights:0"ssd_300_vgg/conv8_2/weights/Assign"ssd_300_vgg/conv8_2/weights/read:028ssd_300_vgg/conv8_2/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv8_2/biases:0!ssd_300_vgg/conv8_2/biases/Assign!ssd_300_vgg/conv8_2/biases/read:02.ssd_300_vgg/conv8_2/biases/Initializer/zeros:08
?
ssd_300_vgg/conv9_1/weights:0"ssd_300_vgg/conv9_1/weights/Assign"ssd_300_vgg/conv9_1/weights/read:028ssd_300_vgg/conv9_1/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv9_1/biases:0!ssd_300_vgg/conv9_1/biases/Assign!ssd_300_vgg/conv9_1/biases/read:02.ssd_300_vgg/conv9_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv9_2/weights:0"ssd_300_vgg/conv9_2/weights/Assign"ssd_300_vgg/conv9_2/weights/read:028ssd_300_vgg/conv9_2/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv9_2/biases:0!ssd_300_vgg/conv9_2/biases/Assign!ssd_300_vgg/conv9_2/biases/read:02.ssd_300_vgg/conv9_2/biases/Initializer/zeros:08
?
ssd_300_vgg/conv10_1/weights:0#ssd_300_vgg/conv10_1/weights/Assign#ssd_300_vgg/conv10_1/weights/read:029ssd_300_vgg/conv10_1/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv10_1/biases:0"ssd_300_vgg/conv10_1/biases/Assign"ssd_300_vgg/conv10_1/biases/read:02/ssd_300_vgg/conv10_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv10_2/weights:0#ssd_300_vgg/conv10_2/weights/Assign#ssd_300_vgg/conv10_2/weights/read:029ssd_300_vgg/conv10_2/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv10_2/biases:0"ssd_300_vgg/conv10_2/biases/Assign"ssd_300_vgg/conv10_2/biases/read:02/ssd_300_vgg/conv10_2/biases/Initializer/zeros:08
?
.ssd_300_vgg/block4_box/L2Normalization/gamma:03ssd_300_vgg/block4_box/L2Normalization/gamma/Assign3ssd_300_vgg/block4_box/L2Normalization/gamma/read:02?ssd_300_vgg/block4_box/L2Normalization/gamma/Initializer/ones:08
?
)ssd_300_vgg/block4_box/conv_loc/weights:0.ssd_300_vgg/block4_box/conv_loc/weights/Assign.ssd_300_vgg/block4_box/conv_loc/weights/read:02Dssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block4_box/conv_loc/biases:0-ssd_300_vgg/block4_box/conv_loc/biases/Assign-ssd_300_vgg/block4_box/conv_loc/biases/read:02:ssd_300_vgg/block4_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block4_box/conv_cls/weights:0.ssd_300_vgg/block4_box/conv_cls/weights/Assign.ssd_300_vgg/block4_box/conv_cls/weights/read:02Dssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block4_box/conv_cls/biases:0-ssd_300_vgg/block4_box/conv_cls/biases/Assign-ssd_300_vgg/block4_box/conv_cls/biases/read:02:ssd_300_vgg/block4_box/conv_cls/biases/Initializer/zeros:08
?
)ssd_300_vgg/block7_box/conv_loc/weights:0.ssd_300_vgg/block7_box/conv_loc/weights/Assign.ssd_300_vgg/block7_box/conv_loc/weights/read:02Dssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block7_box/conv_loc/biases:0-ssd_300_vgg/block7_box/conv_loc/biases/Assign-ssd_300_vgg/block7_box/conv_loc/biases/read:02:ssd_300_vgg/block7_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block7_box/conv_cls/weights:0.ssd_300_vgg/block7_box/conv_cls/weights/Assign.ssd_300_vgg/block7_box/conv_cls/weights/read:02Dssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block7_box/conv_cls/biases:0-ssd_300_vgg/block7_box/conv_cls/biases/Assign-ssd_300_vgg/block7_box/conv_cls/biases/read:02:ssd_300_vgg/block7_box/conv_cls/biases/Initializer/zeros:08
?
)ssd_300_vgg/block8_box/conv_loc/weights:0.ssd_300_vgg/block8_box/conv_loc/weights/Assign.ssd_300_vgg/block8_box/conv_loc/weights/read:02Dssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block8_box/conv_loc/biases:0-ssd_300_vgg/block8_box/conv_loc/biases/Assign-ssd_300_vgg/block8_box/conv_loc/biases/read:02:ssd_300_vgg/block8_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block8_box/conv_cls/weights:0.ssd_300_vgg/block8_box/conv_cls/weights/Assign.ssd_300_vgg/block8_box/conv_cls/weights/read:02Dssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block8_box/conv_cls/biases:0-ssd_300_vgg/block8_box/conv_cls/biases/Assign-ssd_300_vgg/block8_box/conv_cls/biases/read:02:ssd_300_vgg/block8_box/conv_cls/biases/Initializer/zeros:08
?
)ssd_300_vgg/block9_box/conv_loc/weights:0.ssd_300_vgg/block9_box/conv_loc/weights/Assign.ssd_300_vgg/block9_box/conv_loc/weights/read:02Dssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block9_box/conv_loc/biases:0-ssd_300_vgg/block9_box/conv_loc/biases/Assign-ssd_300_vgg/block9_box/conv_loc/biases/read:02:ssd_300_vgg/block9_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block9_box/conv_cls/weights:0.ssd_300_vgg/block9_box/conv_cls/weights/Assign.ssd_300_vgg/block9_box/conv_cls/weights/read:02Dssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block9_box/conv_cls/biases:0-ssd_300_vgg/block9_box/conv_cls/biases/Assign-ssd_300_vgg/block9_box/conv_cls/biases/read:02:ssd_300_vgg/block9_box/conv_cls/biases/Initializer/zeros:08
?
*ssd_300_vgg/block10_box/conv_loc/weights:0/ssd_300_vgg/block10_box/conv_loc/weights/Assign/ssd_300_vgg/block10_box/conv_loc/weights/read:02Essd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block10_box/conv_loc/biases:0.ssd_300_vgg/block10_box/conv_loc/biases/Assign.ssd_300_vgg/block10_box/conv_loc/biases/read:02;ssd_300_vgg/block10_box/conv_loc/biases/Initializer/zeros:08
?
*ssd_300_vgg/block10_box/conv_cls/weights:0/ssd_300_vgg/block10_box/conv_cls/weights/Assign/ssd_300_vgg/block10_box/conv_cls/weights/read:02Essd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block10_box/conv_cls/biases:0.ssd_300_vgg/block10_box/conv_cls/biases/Assign.ssd_300_vgg/block10_box/conv_cls/biases/read:02;ssd_300_vgg/block10_box/conv_cls/biases/Initializer/zeros:08
?
*ssd_300_vgg/block11_box/conv_loc/weights:0/ssd_300_vgg/block11_box/conv_loc/weights/Assign/ssd_300_vgg/block11_box/conv_loc/weights/read:02Essd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block11_box/conv_loc/biases:0.ssd_300_vgg/block11_box/conv_loc/biases/Assign.ssd_300_vgg/block11_box/conv_loc/biases/read:02;ssd_300_vgg/block11_box/conv_loc/biases/Initializer/zeros:08
?
*ssd_300_vgg/block11_box/conv_cls/weights:0/ssd_300_vgg/block11_box/conv_cls/weights/Assign/ssd_300_vgg/block11_box/conv_cls/weights/read:02Essd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block11_box/conv_cls/biases:0.ssd_300_vgg/block11_box/conv_cls/biases/Assign.ssd_300_vgg/block11_box/conv_cls/biases/read:02;ssd_300_vgg/block11_box/conv_cls/biases/Initializer/zeros:08"??
trainable_variables????
?
ssd_300_vgg/conv_init/weights:0$ssd_300_vgg/conv_init/weights/Assign$ssd_300_vgg/conv_init/weights/read:02:ssd_300_vgg/conv_init/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv_init/biases:0#ssd_300_vgg/conv_init/biases/Assign#ssd_300_vgg/conv_init/biases/read:020ssd_300_vgg/conv_init/biases/Initializer/zeros:08
?
 ssd_300_vgg/batch_norm_00/beta:0%ssd_300_vgg/batch_norm_00/beta/Assign%ssd_300_vgg/batch_norm_00/beta/read:022ssd_300_vgg/batch_norm_00/beta/Initializer/zeros:08
?
!ssd_300_vgg/batch_norm_00/gamma:0&ssd_300_vgg/batch_norm_00/gamma/Assign&ssd_300_vgg/batch_norm_00/gamma/read:022ssd_300_vgg/batch_norm_00/gamma/Initializer/ones:08
?
!ssd_300_vgg/conv_init_1/weights:0&ssd_300_vgg/conv_init_1/weights/Assign&ssd_300_vgg/conv_init_1/weights/read:02<ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform:08
?
 ssd_300_vgg/conv_init_1/biases:0%ssd_300_vgg/conv_init_1/biases/Assign%ssd_300_vgg/conv_init_1/biases/read:022ssd_300_vgg/conv_init_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_0/batch_norm_0/beta:00ssd_300_vgg/resblock0_0/batch_norm_0/beta/Assign0ssd_300_vgg/resblock0_0/batch_norm_0/beta/read:02=ssd_300_vgg/resblock0_0/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_0/batch_norm_0/gamma:01ssd_300_vgg/resblock0_0/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock0_0/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock0_0/batch_norm_0/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock0_0/conv_0/weights:0-ssd_300_vgg/resblock0_0/conv_0/weights/Assign-ssd_300_vgg/resblock0_0/conv_0/weights/read:02Cssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_0/conv_0/biases:0,ssd_300_vgg/resblock0_0/conv_0/biases/Assign,ssd_300_vgg/resblock0_0/conv_0/biases/read:029ssd_300_vgg/resblock0_0/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_0/batch_norm_1/beta:00ssd_300_vgg/resblock0_0/batch_norm_1/beta/Assign0ssd_300_vgg/resblock0_0/batch_norm_1/beta/read:02=ssd_300_vgg/resblock0_0/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_0/batch_norm_1/gamma:01ssd_300_vgg/resblock0_0/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock0_0/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock0_0/batch_norm_1/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock0_0/conv_1/weights:0-ssd_300_vgg/resblock0_0/conv_1/weights/Assign-ssd_300_vgg/resblock0_0/conv_1/weights/read:02Cssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_0/conv_1/biases:0,ssd_300_vgg/resblock0_0/conv_1/biases/Assign,ssd_300_vgg/resblock0_0/conv_1/biases/read:029ssd_300_vgg/resblock0_0/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_1/batch_norm_0/beta:00ssd_300_vgg/resblock0_1/batch_norm_0/beta/Assign0ssd_300_vgg/resblock0_1/batch_norm_0/beta/read:02=ssd_300_vgg/resblock0_1/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_1/batch_norm_0/gamma:01ssd_300_vgg/resblock0_1/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock0_1/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock0_1/batch_norm_0/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock0_1/conv_0/weights:0-ssd_300_vgg/resblock0_1/conv_0/weights/Assign-ssd_300_vgg/resblock0_1/conv_0/weights/read:02Cssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_1/conv_0/biases:0,ssd_300_vgg/resblock0_1/conv_0/biases/Assign,ssd_300_vgg/resblock0_1/conv_0/biases/read:029ssd_300_vgg/resblock0_1/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_1/batch_norm_1/beta:00ssd_300_vgg/resblock0_1/batch_norm_1/beta/Assign0ssd_300_vgg/resblock0_1/batch_norm_1/beta/read:02=ssd_300_vgg/resblock0_1/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_1/batch_norm_1/gamma:01ssd_300_vgg/resblock0_1/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock0_1/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock0_1/batch_norm_1/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock0_1/conv_1/weights:0-ssd_300_vgg/resblock0_1/conv_1/weights/Assign-ssd_300_vgg/resblock0_1/conv_1/weights/read:02Cssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_1/conv_1/biases:0,ssd_300_vgg/resblock0_1/conv_1/biases/Assign,ssd_300_vgg/resblock0_1/conv_1/biases/read:029ssd_300_vgg/resblock0_1/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_0/batch_norm_0/beta:00ssd_300_vgg/resblock1_0/batch_norm_0/beta/Assign0ssd_300_vgg/resblock1_0/batch_norm_0/beta/read:02=ssd_300_vgg/resblock1_0/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_0/batch_norm_0/gamma:01ssd_300_vgg/resblock1_0/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock1_0/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock1_0/batch_norm_0/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock1_0/conv_0/weights:0-ssd_300_vgg/resblock1_0/conv_0/weights/Assign-ssd_300_vgg/resblock1_0/conv_0/weights/read:02Cssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_0/conv_0/biases:0,ssd_300_vgg/resblock1_0/conv_0/biases/Assign,ssd_300_vgg/resblock1_0/conv_0/biases/read:029ssd_300_vgg/resblock1_0/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_0/conv_init/weights:00ssd_300_vgg/resblock1_0/conv_init/weights/Assign0ssd_300_vgg/resblock1_0/conv_init/weights/read:02Fssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform:08
?
*ssd_300_vgg/resblock1_0/conv_init/biases:0/ssd_300_vgg/resblock1_0/conv_init/biases/Assign/ssd_300_vgg/resblock1_0/conv_init/biases/read:02<ssd_300_vgg/resblock1_0/conv_init/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_0/batch_norm_1/beta:00ssd_300_vgg/resblock1_0/batch_norm_1/beta/Assign0ssd_300_vgg/resblock1_0/batch_norm_1/beta/read:02=ssd_300_vgg/resblock1_0/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_0/batch_norm_1/gamma:01ssd_300_vgg/resblock1_0/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock1_0/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock1_0/batch_norm_1/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock1_0/conv_1/weights:0-ssd_300_vgg/resblock1_0/conv_1/weights/Assign-ssd_300_vgg/resblock1_0/conv_1/weights/read:02Cssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_0/conv_1/biases:0,ssd_300_vgg/resblock1_0/conv_1/biases/Assign,ssd_300_vgg/resblock1_0/conv_1/biases/read:029ssd_300_vgg/resblock1_0/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_1/batch_norm_0/beta:00ssd_300_vgg/resblock1_1/batch_norm_0/beta/Assign0ssd_300_vgg/resblock1_1/batch_norm_0/beta/read:02=ssd_300_vgg/resblock1_1/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_1/batch_norm_0/gamma:01ssd_300_vgg/resblock1_1/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock1_1/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock1_1/batch_norm_0/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock1_1/conv_0/weights:0-ssd_300_vgg/resblock1_1/conv_0/weights/Assign-ssd_300_vgg/resblock1_1/conv_0/weights/read:02Cssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_1/conv_0/biases:0,ssd_300_vgg/resblock1_1/conv_0/biases/Assign,ssd_300_vgg/resblock1_1/conv_0/biases/read:029ssd_300_vgg/resblock1_1/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_1/batch_norm_1/beta:00ssd_300_vgg/resblock1_1/batch_norm_1/beta/Assign0ssd_300_vgg/resblock1_1/batch_norm_1/beta/read:02=ssd_300_vgg/resblock1_1/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_1/batch_norm_1/gamma:01ssd_300_vgg/resblock1_1/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock1_1/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock1_1/batch_norm_1/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock1_1/conv_1/weights:0-ssd_300_vgg/resblock1_1/conv_1/weights/Assign-ssd_300_vgg/resblock1_1/conv_1/weights/read:02Cssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_1/conv_1/biases:0,ssd_300_vgg/resblock1_1/conv_1/biases/Assign,ssd_300_vgg/resblock1_1/conv_1/biases/read:029ssd_300_vgg/resblock1_1/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_0/batch_norm_0/beta:00ssd_300_vgg/resblock2_0/batch_norm_0/beta/Assign0ssd_300_vgg/resblock2_0/batch_norm_0/beta/read:02=ssd_300_vgg/resblock2_0/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_0/batch_norm_0/gamma:01ssd_300_vgg/resblock2_0/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock2_0/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock2_0/batch_norm_0/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock2_0/conv_0/weights:0-ssd_300_vgg/resblock2_0/conv_0/weights/Assign-ssd_300_vgg/resblock2_0/conv_0/weights/read:02Cssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_0/conv_0/biases:0,ssd_300_vgg/resblock2_0/conv_0/biases/Assign,ssd_300_vgg/resblock2_0/conv_0/biases/read:029ssd_300_vgg/resblock2_0/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_0/conv_init/weights:00ssd_300_vgg/resblock2_0/conv_init/weights/Assign0ssd_300_vgg/resblock2_0/conv_init/weights/read:02Fssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform:08
?
*ssd_300_vgg/resblock2_0/conv_init/biases:0/ssd_300_vgg/resblock2_0/conv_init/biases/Assign/ssd_300_vgg/resblock2_0/conv_init/biases/read:02<ssd_300_vgg/resblock2_0/conv_init/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_0/batch_norm_1/beta:00ssd_300_vgg/resblock2_0/batch_norm_1/beta/Assign0ssd_300_vgg/resblock2_0/batch_norm_1/beta/read:02=ssd_300_vgg/resblock2_0/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_0/batch_norm_1/gamma:01ssd_300_vgg/resblock2_0/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock2_0/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock2_0/batch_norm_1/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock2_0/conv_1/weights:0-ssd_300_vgg/resblock2_0/conv_1/weights/Assign-ssd_300_vgg/resblock2_0/conv_1/weights/read:02Cssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_0/conv_1/biases:0,ssd_300_vgg/resblock2_0/conv_1/biases/Assign,ssd_300_vgg/resblock2_0/conv_1/biases/read:029ssd_300_vgg/resblock2_0/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_1/batch_norm_0/beta:00ssd_300_vgg/resblock2_1/batch_norm_0/beta/Assign0ssd_300_vgg/resblock2_1/batch_norm_0/beta/read:02=ssd_300_vgg/resblock2_1/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_1/batch_norm_0/gamma:01ssd_300_vgg/resblock2_1/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock2_1/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock2_1/batch_norm_0/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock2_1/conv_0/weights:0-ssd_300_vgg/resblock2_1/conv_0/weights/Assign-ssd_300_vgg/resblock2_1/conv_0/weights/read:02Cssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_1/conv_0/biases:0,ssd_300_vgg/resblock2_1/conv_0/biases/Assign,ssd_300_vgg/resblock2_1/conv_0/biases/read:029ssd_300_vgg/resblock2_1/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_1/batch_norm_1/beta:00ssd_300_vgg/resblock2_1/batch_norm_1/beta/Assign0ssd_300_vgg/resblock2_1/batch_norm_1/beta/read:02=ssd_300_vgg/resblock2_1/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_1/batch_norm_1/gamma:01ssd_300_vgg/resblock2_1/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock2_1/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock2_1/batch_norm_1/gamma/Initializer/ones:08
?
(ssd_300_vgg/resblock2_1/conv_1/weights:0-ssd_300_vgg/resblock2_1/conv_1/weights/Assign-ssd_300_vgg/resblock2_1/conv_1/weights/read:02Cssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_1/conv_1/biases:0,ssd_300_vgg/resblock2_1/conv_1/biases/Assign,ssd_300_vgg/resblock2_1/conv_1/biases/read:029ssd_300_vgg/resblock2_1/conv_1/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_0/batch_norm_0/beta:01ssd_300_vgg/resblock_3_0/batch_norm_0/beta/Assign1ssd_300_vgg/resblock_3_0/batch_norm_0/beta/read:02>ssd_300_vgg/resblock_3_0/batch_norm_0/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_0/batch_norm_0/gamma:02ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/Assign2ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/read:02>ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/Initializer/ones:08
?
)ssd_300_vgg/resblock_3_0/conv_0/weights:0.ssd_300_vgg/resblock_3_0/conv_0/weights/Assign.ssd_300_vgg/resblock_3_0/conv_0/weights/read:02Dssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_0/conv_0/biases:0-ssd_300_vgg/resblock_3_0/conv_0/biases/Assign-ssd_300_vgg/resblock_3_0/conv_0/biases/read:02:ssd_300_vgg/resblock_3_0/conv_0/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_0/conv_init/weights:01ssd_300_vgg/resblock_3_0/conv_init/weights/Assign1ssd_300_vgg/resblock_3_0/conv_init/weights/read:02Gssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform:08
?
+ssd_300_vgg/resblock_3_0/conv_init/biases:00ssd_300_vgg/resblock_3_0/conv_init/biases/Assign0ssd_300_vgg/resblock_3_0/conv_init/biases/read:02=ssd_300_vgg/resblock_3_0/conv_init/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_0/batch_norm_1/beta:01ssd_300_vgg/resblock_3_0/batch_norm_1/beta/Assign1ssd_300_vgg/resblock_3_0/batch_norm_1/beta/read:02>ssd_300_vgg/resblock_3_0/batch_norm_1/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_0/batch_norm_1/gamma:02ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/Assign2ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/read:02>ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/Initializer/ones:08
?
)ssd_300_vgg/resblock_3_0/conv_1/weights:0.ssd_300_vgg/resblock_3_0/conv_1/weights/Assign.ssd_300_vgg/resblock_3_0/conv_1/weights/read:02Dssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_0/conv_1/biases:0-ssd_300_vgg/resblock_3_0/conv_1/biases/Assign-ssd_300_vgg/resblock_3_0/conv_1/biases/read:02:ssd_300_vgg/resblock_3_0/conv_1/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_1/batch_norm_0/beta:01ssd_300_vgg/resblock_3_1/batch_norm_0/beta/Assign1ssd_300_vgg/resblock_3_1/batch_norm_0/beta/read:02>ssd_300_vgg/resblock_3_1/batch_norm_0/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_1/batch_norm_0/gamma:02ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/Assign2ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/read:02>ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/Initializer/ones:08
?
)ssd_300_vgg/resblock_3_1/conv_0/weights:0.ssd_300_vgg/resblock_3_1/conv_0/weights/Assign.ssd_300_vgg/resblock_3_1/conv_0/weights/read:02Dssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_1/conv_0/biases:0-ssd_300_vgg/resblock_3_1/conv_0/biases/Assign-ssd_300_vgg/resblock_3_1/conv_0/biases/read:02:ssd_300_vgg/resblock_3_1/conv_0/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_1/batch_norm_1/beta:01ssd_300_vgg/resblock_3_1/batch_norm_1/beta/Assign1ssd_300_vgg/resblock_3_1/batch_norm_1/beta/read:02>ssd_300_vgg/resblock_3_1/batch_norm_1/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_1/batch_norm_1/gamma:02ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/Assign2ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/read:02>ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/Initializer/ones:08
?
)ssd_300_vgg/resblock_3_1/conv_1/weights:0.ssd_300_vgg/resblock_3_1/conv_1/weights/Assign.ssd_300_vgg/resblock_3_1/conv_1/weights/read:02Dssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_1/conv_1/biases:0-ssd_300_vgg/resblock_3_1/conv_1/biases/Assign-ssd_300_vgg/resblock_3_1/conv_1/biases/read:02:ssd_300_vgg/resblock_3_1/conv_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv8_1/weights:0"ssd_300_vgg/conv8_1/weights/Assign"ssd_300_vgg/conv8_1/weights/read:028ssd_300_vgg/conv8_1/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv8_1/biases:0!ssd_300_vgg/conv8_1/biases/Assign!ssd_300_vgg/conv8_1/biases/read:02.ssd_300_vgg/conv8_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv8_2/weights:0"ssd_300_vgg/conv8_2/weights/Assign"ssd_300_vgg/conv8_2/weights/read:028ssd_300_vgg/conv8_2/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv8_2/biases:0!ssd_300_vgg/conv8_2/biases/Assign!ssd_300_vgg/conv8_2/biases/read:02.ssd_300_vgg/conv8_2/biases/Initializer/zeros:08
?
ssd_300_vgg/conv9_1/weights:0"ssd_300_vgg/conv9_1/weights/Assign"ssd_300_vgg/conv9_1/weights/read:028ssd_300_vgg/conv9_1/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv9_1/biases:0!ssd_300_vgg/conv9_1/biases/Assign!ssd_300_vgg/conv9_1/biases/read:02.ssd_300_vgg/conv9_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv9_2/weights:0"ssd_300_vgg/conv9_2/weights/Assign"ssd_300_vgg/conv9_2/weights/read:028ssd_300_vgg/conv9_2/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv9_2/biases:0!ssd_300_vgg/conv9_2/biases/Assign!ssd_300_vgg/conv9_2/biases/read:02.ssd_300_vgg/conv9_2/biases/Initializer/zeros:08
?
ssd_300_vgg/conv10_1/weights:0#ssd_300_vgg/conv10_1/weights/Assign#ssd_300_vgg/conv10_1/weights/read:029ssd_300_vgg/conv10_1/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv10_1/biases:0"ssd_300_vgg/conv10_1/biases/Assign"ssd_300_vgg/conv10_1/biases/read:02/ssd_300_vgg/conv10_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv10_2/weights:0#ssd_300_vgg/conv10_2/weights/Assign#ssd_300_vgg/conv10_2/weights/read:029ssd_300_vgg/conv10_2/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv10_2/biases:0"ssd_300_vgg/conv10_2/biases/Assign"ssd_300_vgg/conv10_2/biases/read:02/ssd_300_vgg/conv10_2/biases/Initializer/zeros:08
?
.ssd_300_vgg/block4_box/L2Normalization/gamma:03ssd_300_vgg/block4_box/L2Normalization/gamma/Assign3ssd_300_vgg/block4_box/L2Normalization/gamma/read:02?ssd_300_vgg/block4_box/L2Normalization/gamma/Initializer/ones:08
?
)ssd_300_vgg/block4_box/conv_loc/weights:0.ssd_300_vgg/block4_box/conv_loc/weights/Assign.ssd_300_vgg/block4_box/conv_loc/weights/read:02Dssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block4_box/conv_loc/biases:0-ssd_300_vgg/block4_box/conv_loc/biases/Assign-ssd_300_vgg/block4_box/conv_loc/biases/read:02:ssd_300_vgg/block4_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block4_box/conv_cls/weights:0.ssd_300_vgg/block4_box/conv_cls/weights/Assign.ssd_300_vgg/block4_box/conv_cls/weights/read:02Dssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block4_box/conv_cls/biases:0-ssd_300_vgg/block4_box/conv_cls/biases/Assign-ssd_300_vgg/block4_box/conv_cls/biases/read:02:ssd_300_vgg/block4_box/conv_cls/biases/Initializer/zeros:08
?
)ssd_300_vgg/block7_box/conv_loc/weights:0.ssd_300_vgg/block7_box/conv_loc/weights/Assign.ssd_300_vgg/block7_box/conv_loc/weights/read:02Dssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block7_box/conv_loc/biases:0-ssd_300_vgg/block7_box/conv_loc/biases/Assign-ssd_300_vgg/block7_box/conv_loc/biases/read:02:ssd_300_vgg/block7_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block7_box/conv_cls/weights:0.ssd_300_vgg/block7_box/conv_cls/weights/Assign.ssd_300_vgg/block7_box/conv_cls/weights/read:02Dssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block7_box/conv_cls/biases:0-ssd_300_vgg/block7_box/conv_cls/biases/Assign-ssd_300_vgg/block7_box/conv_cls/biases/read:02:ssd_300_vgg/block7_box/conv_cls/biases/Initializer/zeros:08
?
)ssd_300_vgg/block8_box/conv_loc/weights:0.ssd_300_vgg/block8_box/conv_loc/weights/Assign.ssd_300_vgg/block8_box/conv_loc/weights/read:02Dssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block8_box/conv_loc/biases:0-ssd_300_vgg/block8_box/conv_loc/biases/Assign-ssd_300_vgg/block8_box/conv_loc/biases/read:02:ssd_300_vgg/block8_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block8_box/conv_cls/weights:0.ssd_300_vgg/block8_box/conv_cls/weights/Assign.ssd_300_vgg/block8_box/conv_cls/weights/read:02Dssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block8_box/conv_cls/biases:0-ssd_300_vgg/block8_box/conv_cls/biases/Assign-ssd_300_vgg/block8_box/conv_cls/biases/read:02:ssd_300_vgg/block8_box/conv_cls/biases/Initializer/zeros:08
?
)ssd_300_vgg/block9_box/conv_loc/weights:0.ssd_300_vgg/block9_box/conv_loc/weights/Assign.ssd_300_vgg/block9_box/conv_loc/weights/read:02Dssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block9_box/conv_loc/biases:0-ssd_300_vgg/block9_box/conv_loc/biases/Assign-ssd_300_vgg/block9_box/conv_loc/biases/read:02:ssd_300_vgg/block9_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block9_box/conv_cls/weights:0.ssd_300_vgg/block9_box/conv_cls/weights/Assign.ssd_300_vgg/block9_box/conv_cls/weights/read:02Dssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block9_box/conv_cls/biases:0-ssd_300_vgg/block9_box/conv_cls/biases/Assign-ssd_300_vgg/block9_box/conv_cls/biases/read:02:ssd_300_vgg/block9_box/conv_cls/biases/Initializer/zeros:08
?
*ssd_300_vgg/block10_box/conv_loc/weights:0/ssd_300_vgg/block10_box/conv_loc/weights/Assign/ssd_300_vgg/block10_box/conv_loc/weights/read:02Essd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block10_box/conv_loc/biases:0.ssd_300_vgg/block10_box/conv_loc/biases/Assign.ssd_300_vgg/block10_box/conv_loc/biases/read:02;ssd_300_vgg/block10_box/conv_loc/biases/Initializer/zeros:08
?
*ssd_300_vgg/block10_box/conv_cls/weights:0/ssd_300_vgg/block10_box/conv_cls/weights/Assign/ssd_300_vgg/block10_box/conv_cls/weights/read:02Essd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block10_box/conv_cls/biases:0.ssd_300_vgg/block10_box/conv_cls/biases/Assign.ssd_300_vgg/block10_box/conv_cls/biases/read:02;ssd_300_vgg/block10_box/conv_cls/biases/Initializer/zeros:08
?
*ssd_300_vgg/block11_box/conv_loc/weights:0/ssd_300_vgg/block11_box/conv_loc/weights/Assign/ssd_300_vgg/block11_box/conv_loc/weights/read:02Essd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block11_box/conv_loc/biases:0.ssd_300_vgg/block11_box/conv_loc/biases/Assign.ssd_300_vgg/block11_box/conv_loc/biases/read:02;ssd_300_vgg/block11_box/conv_loc/biases/Initializer/zeros:08
?
*ssd_300_vgg/block11_box/conv_cls/weights:0/ssd_300_vgg/block11_box/conv_cls/weights/Assign/ssd_300_vgg/block11_box/conv_cls/weights/read:02Essd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block11_box/conv_cls/biases:0.ssd_300_vgg/block11_box/conv_cls/biases/Assign.ssd_300_vgg/block11_box/conv_cls/biases/read:02;ssd_300_vgg/block11_box/conv_cls/biases/Initializer/zeros:08"??
	variables????
?
ssd_300_vgg/conv_init/weights:0$ssd_300_vgg/conv_init/weights/Assign$ssd_300_vgg/conv_init/weights/read:02:ssd_300_vgg/conv_init/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv_init/biases:0#ssd_300_vgg/conv_init/biases/Assign#ssd_300_vgg/conv_init/biases/read:020ssd_300_vgg/conv_init/biases/Initializer/zeros:08
?
 ssd_300_vgg/batch_norm_00/beta:0%ssd_300_vgg/batch_norm_00/beta/Assign%ssd_300_vgg/batch_norm_00/beta/read:022ssd_300_vgg/batch_norm_00/beta/Initializer/zeros:08
?
!ssd_300_vgg/batch_norm_00/gamma:0&ssd_300_vgg/batch_norm_00/gamma/Assign&ssd_300_vgg/batch_norm_00/gamma/read:022ssd_300_vgg/batch_norm_00/gamma/Initializer/ones:08
?
'ssd_300_vgg/batch_norm_00/moving_mean:0,ssd_300_vgg/batch_norm_00/moving_mean/Assign,ssd_300_vgg/batch_norm_00/moving_mean/read:029ssd_300_vgg/batch_norm_00/moving_mean/Initializer/zeros:0
?
+ssd_300_vgg/batch_norm_00/moving_variance:00ssd_300_vgg/batch_norm_00/moving_variance/Assign0ssd_300_vgg/batch_norm_00/moving_variance/read:02<ssd_300_vgg/batch_norm_00/moving_variance/Initializer/ones:0
?
!ssd_300_vgg/conv_init_1/weights:0&ssd_300_vgg/conv_init_1/weights/Assign&ssd_300_vgg/conv_init_1/weights/read:02<ssd_300_vgg/conv_init_1/weights/Initializer/random_uniform:08
?
 ssd_300_vgg/conv_init_1/biases:0%ssd_300_vgg/conv_init_1/biases/Assign%ssd_300_vgg/conv_init_1/biases/read:022ssd_300_vgg/conv_init_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_0/batch_norm_0/beta:00ssd_300_vgg/resblock0_0/batch_norm_0/beta/Assign0ssd_300_vgg/resblock0_0/batch_norm_0/beta/read:02=ssd_300_vgg/resblock0_0/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_0/batch_norm_0/gamma:01ssd_300_vgg/resblock0_0/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock0_0/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock0_0/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean:07ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock0_0/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock0_0/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock0_0/conv_0/weights:0-ssd_300_vgg/resblock0_0/conv_0/weights/Assign-ssd_300_vgg/resblock0_0/conv_0/weights/read:02Cssd_300_vgg/resblock0_0/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_0/conv_0/biases:0,ssd_300_vgg/resblock0_0/conv_0/biases/Assign,ssd_300_vgg/resblock0_0/conv_0/biases/read:029ssd_300_vgg/resblock0_0/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_0/batch_norm_1/beta:00ssd_300_vgg/resblock0_0/batch_norm_1/beta/Assign0ssd_300_vgg/resblock0_0/batch_norm_1/beta/read:02=ssd_300_vgg/resblock0_0/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_0/batch_norm_1/gamma:01ssd_300_vgg/resblock0_0/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock0_0/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock0_0/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean:07ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock0_0/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock0_0/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock0_0/conv_1/weights:0-ssd_300_vgg/resblock0_0/conv_1/weights/Assign-ssd_300_vgg/resblock0_0/conv_1/weights/read:02Cssd_300_vgg/resblock0_0/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_0/conv_1/biases:0,ssd_300_vgg/resblock0_0/conv_1/biases/Assign,ssd_300_vgg/resblock0_0/conv_1/biases/read:029ssd_300_vgg/resblock0_0/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_1/batch_norm_0/beta:00ssd_300_vgg/resblock0_1/batch_norm_0/beta/Assign0ssd_300_vgg/resblock0_1/batch_norm_0/beta/read:02=ssd_300_vgg/resblock0_1/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_1/batch_norm_0/gamma:01ssd_300_vgg/resblock0_1/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock0_1/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock0_1/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean:07ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock0_1/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock0_1/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock0_1/conv_0/weights:0-ssd_300_vgg/resblock0_1/conv_0/weights/Assign-ssd_300_vgg/resblock0_1/conv_0/weights/read:02Cssd_300_vgg/resblock0_1/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_1/conv_0/biases:0,ssd_300_vgg/resblock0_1/conv_0/biases/Assign,ssd_300_vgg/resblock0_1/conv_0/biases/read:029ssd_300_vgg/resblock0_1/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock0_1/batch_norm_1/beta:00ssd_300_vgg/resblock0_1/batch_norm_1/beta/Assign0ssd_300_vgg/resblock0_1/batch_norm_1/beta/read:02=ssd_300_vgg/resblock0_1/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock0_1/batch_norm_1/gamma:01ssd_300_vgg/resblock0_1/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock0_1/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock0_1/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean:07ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock0_1/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock0_1/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock0_1/conv_1/weights:0-ssd_300_vgg/resblock0_1/conv_1/weights/Assign-ssd_300_vgg/resblock0_1/conv_1/weights/read:02Cssd_300_vgg/resblock0_1/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock0_1/conv_1/biases:0,ssd_300_vgg/resblock0_1/conv_1/biases/Assign,ssd_300_vgg/resblock0_1/conv_1/biases/read:029ssd_300_vgg/resblock0_1/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_0/batch_norm_0/beta:00ssd_300_vgg/resblock1_0/batch_norm_0/beta/Assign0ssd_300_vgg/resblock1_0/batch_norm_0/beta/read:02=ssd_300_vgg/resblock1_0/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_0/batch_norm_0/gamma:01ssd_300_vgg/resblock1_0/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock1_0/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock1_0/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean:07ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock1_0/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock1_0/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock1_0/conv_0/weights:0-ssd_300_vgg/resblock1_0/conv_0/weights/Assign-ssd_300_vgg/resblock1_0/conv_0/weights/read:02Cssd_300_vgg/resblock1_0/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_0/conv_0/biases:0,ssd_300_vgg/resblock1_0/conv_0/biases/Assign,ssd_300_vgg/resblock1_0/conv_0/biases/read:029ssd_300_vgg/resblock1_0/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_0/conv_init/weights:00ssd_300_vgg/resblock1_0/conv_init/weights/Assign0ssd_300_vgg/resblock1_0/conv_init/weights/read:02Fssd_300_vgg/resblock1_0/conv_init/weights/Initializer/random_uniform:08
?
*ssd_300_vgg/resblock1_0/conv_init/biases:0/ssd_300_vgg/resblock1_0/conv_init/biases/Assign/ssd_300_vgg/resblock1_0/conv_init/biases/read:02<ssd_300_vgg/resblock1_0/conv_init/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_0/batch_norm_1/beta:00ssd_300_vgg/resblock1_0/batch_norm_1/beta/Assign0ssd_300_vgg/resblock1_0/batch_norm_1/beta/read:02=ssd_300_vgg/resblock1_0/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_0/batch_norm_1/gamma:01ssd_300_vgg/resblock1_0/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock1_0/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock1_0/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean:07ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock1_0/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock1_0/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock1_0/conv_1/weights:0-ssd_300_vgg/resblock1_0/conv_1/weights/Assign-ssd_300_vgg/resblock1_0/conv_1/weights/read:02Cssd_300_vgg/resblock1_0/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_0/conv_1/biases:0,ssd_300_vgg/resblock1_0/conv_1/biases/Assign,ssd_300_vgg/resblock1_0/conv_1/biases/read:029ssd_300_vgg/resblock1_0/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_1/batch_norm_0/beta:00ssd_300_vgg/resblock1_1/batch_norm_0/beta/Assign0ssd_300_vgg/resblock1_1/batch_norm_0/beta/read:02=ssd_300_vgg/resblock1_1/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_1/batch_norm_0/gamma:01ssd_300_vgg/resblock1_1/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock1_1/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock1_1/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean:07ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock1_1/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock1_1/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock1_1/conv_0/weights:0-ssd_300_vgg/resblock1_1/conv_0/weights/Assign-ssd_300_vgg/resblock1_1/conv_0/weights/read:02Cssd_300_vgg/resblock1_1/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_1/conv_0/biases:0,ssd_300_vgg/resblock1_1/conv_0/biases/Assign,ssd_300_vgg/resblock1_1/conv_0/biases/read:029ssd_300_vgg/resblock1_1/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock1_1/batch_norm_1/beta:00ssd_300_vgg/resblock1_1/batch_norm_1/beta/Assign0ssd_300_vgg/resblock1_1/batch_norm_1/beta/read:02=ssd_300_vgg/resblock1_1/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock1_1/batch_norm_1/gamma:01ssd_300_vgg/resblock1_1/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock1_1/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock1_1/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean:07ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock1_1/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock1_1/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock1_1/conv_1/weights:0-ssd_300_vgg/resblock1_1/conv_1/weights/Assign-ssd_300_vgg/resblock1_1/conv_1/weights/read:02Cssd_300_vgg/resblock1_1/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock1_1/conv_1/biases:0,ssd_300_vgg/resblock1_1/conv_1/biases/Assign,ssd_300_vgg/resblock1_1/conv_1/biases/read:029ssd_300_vgg/resblock1_1/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_0/batch_norm_0/beta:00ssd_300_vgg/resblock2_0/batch_norm_0/beta/Assign0ssd_300_vgg/resblock2_0/batch_norm_0/beta/read:02=ssd_300_vgg/resblock2_0/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_0/batch_norm_0/gamma:01ssd_300_vgg/resblock2_0/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock2_0/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock2_0/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean:07ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock2_0/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock2_0/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock2_0/conv_0/weights:0-ssd_300_vgg/resblock2_0/conv_0/weights/Assign-ssd_300_vgg/resblock2_0/conv_0/weights/read:02Cssd_300_vgg/resblock2_0/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_0/conv_0/biases:0,ssd_300_vgg/resblock2_0/conv_0/biases/Assign,ssd_300_vgg/resblock2_0/conv_0/biases/read:029ssd_300_vgg/resblock2_0/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_0/conv_init/weights:00ssd_300_vgg/resblock2_0/conv_init/weights/Assign0ssd_300_vgg/resblock2_0/conv_init/weights/read:02Fssd_300_vgg/resblock2_0/conv_init/weights/Initializer/random_uniform:08
?
*ssd_300_vgg/resblock2_0/conv_init/biases:0/ssd_300_vgg/resblock2_0/conv_init/biases/Assign/ssd_300_vgg/resblock2_0/conv_init/biases/read:02<ssd_300_vgg/resblock2_0/conv_init/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_0/batch_norm_1/beta:00ssd_300_vgg/resblock2_0/batch_norm_1/beta/Assign0ssd_300_vgg/resblock2_0/batch_norm_1/beta/read:02=ssd_300_vgg/resblock2_0/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_0/batch_norm_1/gamma:01ssd_300_vgg/resblock2_0/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock2_0/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock2_0/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean:07ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock2_0/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock2_0/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock2_0/conv_1/weights:0-ssd_300_vgg/resblock2_0/conv_1/weights/Assign-ssd_300_vgg/resblock2_0/conv_1/weights/read:02Cssd_300_vgg/resblock2_0/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_0/conv_1/biases:0,ssd_300_vgg/resblock2_0/conv_1/biases/Assign,ssd_300_vgg/resblock2_0/conv_1/biases/read:029ssd_300_vgg/resblock2_0/conv_1/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_1/batch_norm_0/beta:00ssd_300_vgg/resblock2_1/batch_norm_0/beta/Assign0ssd_300_vgg/resblock2_1/batch_norm_0/beta/read:02=ssd_300_vgg/resblock2_1/batch_norm_0/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_1/batch_norm_0/gamma:01ssd_300_vgg/resblock2_1/batch_norm_0/gamma/Assign1ssd_300_vgg/resblock2_1/batch_norm_0/gamma/read:02=ssd_300_vgg/resblock2_1/batch_norm_0/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean:07ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/Assign7ssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/read:02Dssd_300_vgg/resblock2_1/batch_norm_0/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance:0;ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/Assign;ssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/read:02Gssd_300_vgg/resblock2_1/batch_norm_0/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock2_1/conv_0/weights:0-ssd_300_vgg/resblock2_1/conv_0/weights/Assign-ssd_300_vgg/resblock2_1/conv_0/weights/read:02Cssd_300_vgg/resblock2_1/conv_0/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_1/conv_0/biases:0,ssd_300_vgg/resblock2_1/conv_0/biases/Assign,ssd_300_vgg/resblock2_1/conv_0/biases/read:029ssd_300_vgg/resblock2_1/conv_0/biases/Initializer/zeros:08
?
+ssd_300_vgg/resblock2_1/batch_norm_1/beta:00ssd_300_vgg/resblock2_1/batch_norm_1/beta/Assign0ssd_300_vgg/resblock2_1/batch_norm_1/beta/read:02=ssd_300_vgg/resblock2_1/batch_norm_1/beta/Initializer/zeros:08
?
,ssd_300_vgg/resblock2_1/batch_norm_1/gamma:01ssd_300_vgg/resblock2_1/batch_norm_1/gamma/Assign1ssd_300_vgg/resblock2_1/batch_norm_1/gamma/read:02=ssd_300_vgg/resblock2_1/batch_norm_1/gamma/Initializer/ones:08
?
2ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean:07ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/Assign7ssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/read:02Dssd_300_vgg/resblock2_1/batch_norm_1/moving_mean/Initializer/zeros:0
?
6ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance:0;ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/Assign;ssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/read:02Gssd_300_vgg/resblock2_1/batch_norm_1/moving_variance/Initializer/ones:0
?
(ssd_300_vgg/resblock2_1/conv_1/weights:0-ssd_300_vgg/resblock2_1/conv_1/weights/Assign-ssd_300_vgg/resblock2_1/conv_1/weights/read:02Cssd_300_vgg/resblock2_1/conv_1/weights/Initializer/random_uniform:08
?
'ssd_300_vgg/resblock2_1/conv_1/biases:0,ssd_300_vgg/resblock2_1/conv_1/biases/Assign,ssd_300_vgg/resblock2_1/conv_1/biases/read:029ssd_300_vgg/resblock2_1/conv_1/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_0/batch_norm_0/beta:01ssd_300_vgg/resblock_3_0/batch_norm_0/beta/Assign1ssd_300_vgg/resblock_3_0/batch_norm_0/beta/read:02>ssd_300_vgg/resblock_3_0/batch_norm_0/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_0/batch_norm_0/gamma:02ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/Assign2ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/read:02>ssd_300_vgg/resblock_3_0/batch_norm_0/gamma/Initializer/ones:08
?
3ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean:08ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/Assign8ssd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/read:02Essd_300_vgg/resblock_3_0/batch_norm_0/moving_mean/Initializer/zeros:0
?
7ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance:0<ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/Assign<ssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/read:02Hssd_300_vgg/resblock_3_0/batch_norm_0/moving_variance/Initializer/ones:0
?
)ssd_300_vgg/resblock_3_0/conv_0/weights:0.ssd_300_vgg/resblock_3_0/conv_0/weights/Assign.ssd_300_vgg/resblock_3_0/conv_0/weights/read:02Dssd_300_vgg/resblock_3_0/conv_0/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_0/conv_0/biases:0-ssd_300_vgg/resblock_3_0/conv_0/biases/Assign-ssd_300_vgg/resblock_3_0/conv_0/biases/read:02:ssd_300_vgg/resblock_3_0/conv_0/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_0/conv_init/weights:01ssd_300_vgg/resblock_3_0/conv_init/weights/Assign1ssd_300_vgg/resblock_3_0/conv_init/weights/read:02Gssd_300_vgg/resblock_3_0/conv_init/weights/Initializer/random_uniform:08
?
+ssd_300_vgg/resblock_3_0/conv_init/biases:00ssd_300_vgg/resblock_3_0/conv_init/biases/Assign0ssd_300_vgg/resblock_3_0/conv_init/biases/read:02=ssd_300_vgg/resblock_3_0/conv_init/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_0/batch_norm_1/beta:01ssd_300_vgg/resblock_3_0/batch_norm_1/beta/Assign1ssd_300_vgg/resblock_3_0/batch_norm_1/beta/read:02>ssd_300_vgg/resblock_3_0/batch_norm_1/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_0/batch_norm_1/gamma:02ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/Assign2ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/read:02>ssd_300_vgg/resblock_3_0/batch_norm_1/gamma/Initializer/ones:08
?
3ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean:08ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/Assign8ssd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/read:02Essd_300_vgg/resblock_3_0/batch_norm_1/moving_mean/Initializer/zeros:0
?
7ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance:0<ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/Assign<ssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/read:02Hssd_300_vgg/resblock_3_0/batch_norm_1/moving_variance/Initializer/ones:0
?
)ssd_300_vgg/resblock_3_0/conv_1/weights:0.ssd_300_vgg/resblock_3_0/conv_1/weights/Assign.ssd_300_vgg/resblock_3_0/conv_1/weights/read:02Dssd_300_vgg/resblock_3_0/conv_1/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_0/conv_1/biases:0-ssd_300_vgg/resblock_3_0/conv_1/biases/Assign-ssd_300_vgg/resblock_3_0/conv_1/biases/read:02:ssd_300_vgg/resblock_3_0/conv_1/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_1/batch_norm_0/beta:01ssd_300_vgg/resblock_3_1/batch_norm_0/beta/Assign1ssd_300_vgg/resblock_3_1/batch_norm_0/beta/read:02>ssd_300_vgg/resblock_3_1/batch_norm_0/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_1/batch_norm_0/gamma:02ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/Assign2ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/read:02>ssd_300_vgg/resblock_3_1/batch_norm_0/gamma/Initializer/ones:08
?
3ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean:08ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/Assign8ssd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/read:02Essd_300_vgg/resblock_3_1/batch_norm_0/moving_mean/Initializer/zeros:0
?
7ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance:0<ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/Assign<ssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/read:02Hssd_300_vgg/resblock_3_1/batch_norm_0/moving_variance/Initializer/ones:0
?
)ssd_300_vgg/resblock_3_1/conv_0/weights:0.ssd_300_vgg/resblock_3_1/conv_0/weights/Assign.ssd_300_vgg/resblock_3_1/conv_0/weights/read:02Dssd_300_vgg/resblock_3_1/conv_0/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_1/conv_0/biases:0-ssd_300_vgg/resblock_3_1/conv_0/biases/Assign-ssd_300_vgg/resblock_3_1/conv_0/biases/read:02:ssd_300_vgg/resblock_3_1/conv_0/biases/Initializer/zeros:08
?
,ssd_300_vgg/resblock_3_1/batch_norm_1/beta:01ssd_300_vgg/resblock_3_1/batch_norm_1/beta/Assign1ssd_300_vgg/resblock_3_1/batch_norm_1/beta/read:02>ssd_300_vgg/resblock_3_1/batch_norm_1/beta/Initializer/zeros:08
?
-ssd_300_vgg/resblock_3_1/batch_norm_1/gamma:02ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/Assign2ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/read:02>ssd_300_vgg/resblock_3_1/batch_norm_1/gamma/Initializer/ones:08
?
3ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean:08ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/Assign8ssd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/read:02Essd_300_vgg/resblock_3_1/batch_norm_1/moving_mean/Initializer/zeros:0
?
7ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance:0<ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/Assign<ssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/read:02Hssd_300_vgg/resblock_3_1/batch_norm_1/moving_variance/Initializer/ones:0
?
)ssd_300_vgg/resblock_3_1/conv_1/weights:0.ssd_300_vgg/resblock_3_1/conv_1/weights/Assign.ssd_300_vgg/resblock_3_1/conv_1/weights/read:02Dssd_300_vgg/resblock_3_1/conv_1/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/resblock_3_1/conv_1/biases:0-ssd_300_vgg/resblock_3_1/conv_1/biases/Assign-ssd_300_vgg/resblock_3_1/conv_1/biases/read:02:ssd_300_vgg/resblock_3_1/conv_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv8_1/weights:0"ssd_300_vgg/conv8_1/weights/Assign"ssd_300_vgg/conv8_1/weights/read:028ssd_300_vgg/conv8_1/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv8_1/biases:0!ssd_300_vgg/conv8_1/biases/Assign!ssd_300_vgg/conv8_1/biases/read:02.ssd_300_vgg/conv8_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv8_2/weights:0"ssd_300_vgg/conv8_2/weights/Assign"ssd_300_vgg/conv8_2/weights/read:028ssd_300_vgg/conv8_2/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv8_2/biases:0!ssd_300_vgg/conv8_2/biases/Assign!ssd_300_vgg/conv8_2/biases/read:02.ssd_300_vgg/conv8_2/biases/Initializer/zeros:08
?
ssd_300_vgg/conv9_1/weights:0"ssd_300_vgg/conv9_1/weights/Assign"ssd_300_vgg/conv9_1/weights/read:028ssd_300_vgg/conv9_1/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv9_1/biases:0!ssd_300_vgg/conv9_1/biases/Assign!ssd_300_vgg/conv9_1/biases/read:02.ssd_300_vgg/conv9_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv9_2/weights:0"ssd_300_vgg/conv9_2/weights/Assign"ssd_300_vgg/conv9_2/weights/read:028ssd_300_vgg/conv9_2/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv9_2/biases:0!ssd_300_vgg/conv9_2/biases/Assign!ssd_300_vgg/conv9_2/biases/read:02.ssd_300_vgg/conv9_2/biases/Initializer/zeros:08
?
ssd_300_vgg/conv10_1/weights:0#ssd_300_vgg/conv10_1/weights/Assign#ssd_300_vgg/conv10_1/weights/read:029ssd_300_vgg/conv10_1/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv10_1/biases:0"ssd_300_vgg/conv10_1/biases/Assign"ssd_300_vgg/conv10_1/biases/read:02/ssd_300_vgg/conv10_1/biases/Initializer/zeros:08
?
ssd_300_vgg/conv10_2/weights:0#ssd_300_vgg/conv10_2/weights/Assign#ssd_300_vgg/conv10_2/weights/read:029ssd_300_vgg/conv10_2/weights/Initializer/random_uniform:08
?
ssd_300_vgg/conv10_2/biases:0"ssd_300_vgg/conv10_2/biases/Assign"ssd_300_vgg/conv10_2/biases/read:02/ssd_300_vgg/conv10_2/biases/Initializer/zeros:08
?
.ssd_300_vgg/block4_box/L2Normalization/gamma:03ssd_300_vgg/block4_box/L2Normalization/gamma/Assign3ssd_300_vgg/block4_box/L2Normalization/gamma/read:02?ssd_300_vgg/block4_box/L2Normalization/gamma/Initializer/ones:08
?
)ssd_300_vgg/block4_box/conv_loc/weights:0.ssd_300_vgg/block4_box/conv_loc/weights/Assign.ssd_300_vgg/block4_box/conv_loc/weights/read:02Dssd_300_vgg/block4_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block4_box/conv_loc/biases:0-ssd_300_vgg/block4_box/conv_loc/biases/Assign-ssd_300_vgg/block4_box/conv_loc/biases/read:02:ssd_300_vgg/block4_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block4_box/conv_cls/weights:0.ssd_300_vgg/block4_box/conv_cls/weights/Assign.ssd_300_vgg/block4_box/conv_cls/weights/read:02Dssd_300_vgg/block4_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block4_box/conv_cls/biases:0-ssd_300_vgg/block4_box/conv_cls/biases/Assign-ssd_300_vgg/block4_box/conv_cls/biases/read:02:ssd_300_vgg/block4_box/conv_cls/biases/Initializer/zeros:08
?
)ssd_300_vgg/block7_box/conv_loc/weights:0.ssd_300_vgg/block7_box/conv_loc/weights/Assign.ssd_300_vgg/block7_box/conv_loc/weights/read:02Dssd_300_vgg/block7_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block7_box/conv_loc/biases:0-ssd_300_vgg/block7_box/conv_loc/biases/Assign-ssd_300_vgg/block7_box/conv_loc/biases/read:02:ssd_300_vgg/block7_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block7_box/conv_cls/weights:0.ssd_300_vgg/block7_box/conv_cls/weights/Assign.ssd_300_vgg/block7_box/conv_cls/weights/read:02Dssd_300_vgg/block7_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block7_box/conv_cls/biases:0-ssd_300_vgg/block7_box/conv_cls/biases/Assign-ssd_300_vgg/block7_box/conv_cls/biases/read:02:ssd_300_vgg/block7_box/conv_cls/biases/Initializer/zeros:08
?
)ssd_300_vgg/block8_box/conv_loc/weights:0.ssd_300_vgg/block8_box/conv_loc/weights/Assign.ssd_300_vgg/block8_box/conv_loc/weights/read:02Dssd_300_vgg/block8_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block8_box/conv_loc/biases:0-ssd_300_vgg/block8_box/conv_loc/biases/Assign-ssd_300_vgg/block8_box/conv_loc/biases/read:02:ssd_300_vgg/block8_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block8_box/conv_cls/weights:0.ssd_300_vgg/block8_box/conv_cls/weights/Assign.ssd_300_vgg/block8_box/conv_cls/weights/read:02Dssd_300_vgg/block8_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block8_box/conv_cls/biases:0-ssd_300_vgg/block8_box/conv_cls/biases/Assign-ssd_300_vgg/block8_box/conv_cls/biases/read:02:ssd_300_vgg/block8_box/conv_cls/biases/Initializer/zeros:08
?
)ssd_300_vgg/block9_box/conv_loc/weights:0.ssd_300_vgg/block9_box/conv_loc/weights/Assign.ssd_300_vgg/block9_box/conv_loc/weights/read:02Dssd_300_vgg/block9_box/conv_loc/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block9_box/conv_loc/biases:0-ssd_300_vgg/block9_box/conv_loc/biases/Assign-ssd_300_vgg/block9_box/conv_loc/biases/read:02:ssd_300_vgg/block9_box/conv_loc/biases/Initializer/zeros:08
?
)ssd_300_vgg/block9_box/conv_cls/weights:0.ssd_300_vgg/block9_box/conv_cls/weights/Assign.ssd_300_vgg/block9_box/conv_cls/weights/read:02Dssd_300_vgg/block9_box/conv_cls/weights/Initializer/random_uniform:08
?
(ssd_300_vgg/block9_box/conv_cls/biases:0-ssd_300_vgg/block9_box/conv_cls/biases/Assign-ssd_300_vgg/block9_box/conv_cls/biases/read:02:ssd_300_vgg/block9_box/conv_cls/biases/Initializer/zeros:08
?
*ssd_300_vgg/block10_box/conv_loc/weights:0/ssd_300_vgg/block10_box/conv_loc/weights/Assign/ssd_300_vgg/block10_box/conv_loc/weights/read:02Essd_300_vgg/block10_box/conv_loc/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block10_box/conv_loc/biases:0.ssd_300_vgg/block10_box/conv_loc/biases/Assign.ssd_300_vgg/block10_box/conv_loc/biases/read:02;ssd_300_vgg/block10_box/conv_loc/biases/Initializer/zeros:08
?
*ssd_300_vgg/block10_box/conv_cls/weights:0/ssd_300_vgg/block10_box/conv_cls/weights/Assign/ssd_300_vgg/block10_box/conv_cls/weights/read:02Essd_300_vgg/block10_box/conv_cls/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block10_box/conv_cls/biases:0.ssd_300_vgg/block10_box/conv_cls/biases/Assign.ssd_300_vgg/block10_box/conv_cls/biases/read:02;ssd_300_vgg/block10_box/conv_cls/biases/Initializer/zeros:08
?
*ssd_300_vgg/block11_box/conv_loc/weights:0/ssd_300_vgg/block11_box/conv_loc/weights/Assign/ssd_300_vgg/block11_box/conv_loc/weights/read:02Essd_300_vgg/block11_box/conv_loc/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block11_box/conv_loc/biases:0.ssd_300_vgg/block11_box/conv_loc/biases/Assign.ssd_300_vgg/block11_box/conv_loc/biases/read:02;ssd_300_vgg/block11_box/conv_loc/biases/Initializer/zeros:08
?
*ssd_300_vgg/block11_box/conv_cls/weights:0/ssd_300_vgg/block11_box/conv_cls/weights/Assign/ssd_300_vgg/block11_box/conv_cls/weights/read:02Essd_300_vgg/block11_box/conv_cls/weights/Initializer/random_uniform:08
?
)ssd_300_vgg/block11_box/conv_cls/biases:0.ssd_300_vgg/block11_box/conv_cls/biases/Assign.ssd_300_vgg/block11_box/conv_cls/biases/read:02;ssd_300_vgg/block11_box/conv_cls/biases/Initializer/zeros:08*?
serving_default?
4
input_image%
input_image:0??C
ps_5;
!ssd_300_vgg/softmax_5/Reshape_1:0B
ls_0:
 ssd_300_vgg/block4_box/Reshape:0B
ls_2:
 ssd_300_vgg/block8_box/Reshape:0A
ps_09
ssd_300_vgg/softmax/Reshape_1:0C
ps_2;
!ssd_300_vgg/softmax_2/Reshape_1:0C
ps_4;
!ssd_300_vgg/softmax_4/Reshape_1:0C
ps_1;
!ssd_300_vgg/softmax_1/Reshape_1:0C
ls_5;
!ssd_300_vgg/block11_box/Reshape:0B
ls_3:
 ssd_300_vgg/block9_box/Reshape:0C
ps_3;
!ssd_300_vgg/softmax_3/Reshape_1:0C
ls_4;
!ssd_300_vgg/block10_box/Reshape:0B
ls_1:
 ssd_300_vgg/block7_box/Reshape:0tensorflow/serving/predict