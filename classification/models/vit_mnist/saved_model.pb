╬з,
Ж┘
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
*
Erf
x"T
y"T"
Ttype:
2
┐
ExtractImagePatches
images"T
patches"T"
ksizes	list(int)(0"
strides	list(int)(0"
rates	list(int)(0"
Ttype:
2	
""
paddingstring:
SAMEVALID
о
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0И
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
list(type)(0И
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	Р
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.12v2.10.0-76-gfdfc646704c8а║(
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:
*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: 
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: 
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@ *
dtype0
М
layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_2/beta
Е
.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
:@*
dtype0
О
layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_2/gamma
З
/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
:@*
dtype0
░
,transformer_block/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,transformer_block/layer_normalization_1/beta
й
@transformer_block/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp,transformer_block/layer_normalization_1/beta*
_output_shapes
:@*
dtype0
▓
-transformer_block/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-transformer_block/layer_normalization_1/gamma
л
Atransformer_block/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp-transformer_block/layer_normalization_1/gamma*
_output_shapes
:@*
dtype0
м
*transformer_block/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*transformer_block/layer_normalization/beta
е
>transformer_block/layer_normalization/beta/Read/ReadVariableOpReadVariableOp*transformer_block/layer_normalization/beta*
_output_shapes
:@*
dtype0
о
+transformer_block/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+transformer_block/layer_normalization/gamma
з
?transformer_block/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp+transformer_block/layer_normalization/gamma*
_output_shapes
:@*
dtype0
А
dense_embed_dim/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namedense_embed_dim/bias
y
(dense_embed_dim/bias/Read/ReadVariableOpReadVariableOpdense_embed_dim/bias*
_output_shapes
:@*
dtype0
И
dense_embed_dim/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*'
shared_namedense_embed_dim/kernel
Б
*dense_embed_dim/kernel/Read/ReadVariableOpReadVariableOpdense_embed_dim/kernel*
_output_shapes

: @*
dtype0
t
dense_mlp/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_mlp/bias
m
"dense_mlp/bias/Read/ReadVariableOpReadVariableOpdense_mlp/bias*
_output_shapes
: *
dtype0
|
dense_mlp/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *!
shared_namedense_mlp/kernel
u
$dense_mlp/kernel/Read/ReadVariableOpReadVariableOpdense_mlp/kernel*
_output_shapes

:@ *
dtype0
└
4transformer_block/multi_head_self_attention/out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64transformer_block/multi_head_self_attention/out/bias
╣
Htransformer_block/multi_head_self_attention/out/bias/Read/ReadVariableOpReadVariableOp4transformer_block/multi_head_self_attention/out/bias*
_output_shapes
:@*
dtype0
╚
6transformer_block/multi_head_self_attention/out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*G
shared_name86transformer_block/multi_head_self_attention/out/kernel
┴
Jtransformer_block/multi_head_self_attention/out/kernel/Read/ReadVariableOpReadVariableOp6transformer_block/multi_head_self_attention/out/kernel*
_output_shapes

:@@*
dtype0
─
6transformer_block/multi_head_self_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*G
shared_name86transformer_block/multi_head_self_attention/value/bias
╜
Jtransformer_block/multi_head_self_attention/value/bias/Read/ReadVariableOpReadVariableOp6transformer_block/multi_head_self_attention/value/bias*
_output_shapes
:@*
dtype0
╠
8transformer_block/multi_head_self_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*I
shared_name:8transformer_block/multi_head_self_attention/value/kernel
┼
Ltransformer_block/multi_head_self_attention/value/kernel/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/value/kernel*
_output_shapes

:@@*
dtype0
└
4transformer_block/multi_head_self_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64transformer_block/multi_head_self_attention/key/bias
╣
Htransformer_block/multi_head_self_attention/key/bias/Read/ReadVariableOpReadVariableOp4transformer_block/multi_head_self_attention/key/bias*
_output_shapes
:@*
dtype0
╚
6transformer_block/multi_head_self_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*G
shared_name86transformer_block/multi_head_self_attention/key/kernel
┴
Jtransformer_block/multi_head_self_attention/key/kernel/Read/ReadVariableOpReadVariableOp6transformer_block/multi_head_self_attention/key/kernel*
_output_shapes

:@@*
dtype0
─
6transformer_block/multi_head_self_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*G
shared_name86transformer_block/multi_head_self_attention/query/bias
╜
Jtransformer_block/multi_head_self_attention/query/bias/Read/ReadVariableOpReadVariableOp6transformer_block/multi_head_self_attention/query/bias*
_output_shapes
:@*
dtype0
╠
8transformer_block/multi_head_self_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*I
shared_name:8transformer_block/multi_head_self_attention/query/kernel
┼
Ltransformer_block/multi_head_self_attention/query/kernel/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/query/kernel*
_output_shapes

:@@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
r
	class_embVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	class_emb
k
class_emb/Read/ReadVariableOpReadVariableOp	class_emb*"
_output_shapes
:@*
dtype0
n
pos_embVarHandleOp*
_output_shapes
: *
dtype0*
shape:A@*
shared_name	pos_emb
g
pos_emb/Read/ReadVariableOpReadVariableOppos_emb*"
_output_shapes
:A@*
dtype0
К
serving_default_input_1Placeholder*/
_output_shapes
:           *
dtype0*$
shape:           
╤	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/bias	class_embpos_emb+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta8transformer_block/multi_head_self_attention/query/kernel6transformer_block/multi_head_self_attention/query/bias6transformer_block/multi_head_self_attention/key/kernel4transformer_block/multi_head_self_attention/key/bias8transformer_block/multi_head_self_attention/value/kernel6transformer_block/multi_head_self_attention/value/bias6transformer_block/multi_head_self_attention/out/kernel4transformer_block/multi_head_self_attention/out/bias-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betadense_mlp/kerneldense_mlp/biasdense_embed_dim/kerneldense_embed_dim/biaslayer_normalization_2/gammalayer_normalization_2/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_36176

NoOpNoOp
цЙ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*аЙ
valueХЙBСЙ BЙЙ
Ф
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
rescale
	pos_emb

	class_emb

patch_proj

enc_layers
mlp_head

signatures*
╩
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
	24

25*
╩
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
	24

25*
* 
░
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
,trace_0
-trace_1
.trace_2
/trace_3* 
6
0trace_0
1trace_1
2trace_2
3trace_3* 
* 
О
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 
C=
VARIABLE_VALUEpos_emb"pos_emb/.ATTRIBUTES/VARIABLE_VALUE*
GA
VARIABLE_VALUE	class_emb$class_emb/.ATTRIBUTES/VARIABLE_VALUE*
ж
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

kernel
bias*

@0*
Т
Alayer_with_weights-0
Alayer-0
Blayer_with_weights-1
Blayer-1
Clayer-2
Dlayer_with_weights-2
Dlayer-3
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses*

Kserving_default* 
LF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8transformer_block/multi_head_self_attention/query/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_block/multi_head_self_attention/query/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_block/multi_head_self_attention/key/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4transformer_block/multi_head_self_attention/key/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8transformer_block/multi_head_self_attention/value/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_block/multi_head_self_attention/value/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6transformer_block/multi_head_self_attention/out/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE4transformer_block/multi_head_self_attention/out/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEdense_mlp/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_mlp/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEdense_embed_dim/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_embed_dim/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE+transformer_block/layer_normalization/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*transformer_block/layer_normalization/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-transformer_block/layer_normalization_1/gamma'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,transformer_block/layer_normalization_1/beta'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUElayer_normalization_2/gamma'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElayer_normalization_2/beta'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
@2
3*

L0
M1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
С
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 

Strace_0* 

Ttrace_0* 

0
1*

0
1*
* 
У
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 
▐
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
batt
cmlp
d
layernorm1
e
layernorm2
fdropout1
gdropout2*
п
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
naxis
	!gamma
"beta*
ж
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

#kernel
$bias*
е
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_random_generator* 
и
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses

%kernel
&bias*
.
!0
"1
#2
$3
%4
&5*
.
!0
"1
#2
$3
%4
&5*
* 
Ш
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
:
Зtrace_0
Иtrace_1
Йtrace_2
Кtrace_3* 
:
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_3* 
* 
<
П	variables
Р	keras_api

Сtotal

Тcount*
M
У	variables
Ф	keras_api

Хtotal

Цcount
Ч
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
z
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15*
z
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15*
* 
Ш
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

Эtrace_0
Юtrace_1* 

Яtrace_0
аtrace_1* 
▐
б	variables
вtrainable_variables
гregularization_losses
д	keras_api
е__call__
+ж&call_and_return_all_conditional_losses
зquery_dense
и	key_dense
йvalue_dense
кcombine_heads*
Д
лlayer_with_weights-0
лlayer-0
мlayer-1
нlayer_with_weights-1
нlayer-2
оlayer-3
п	variables
░trainable_variables
▒regularization_losses
▓	keras_api
│__call__
+┤&call_and_return_all_conditional_losses*
╢
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
	╗axis
	gamma
beta*
╢
╝	variables
╜trainable_variables
╛regularization_losses
┐	keras_api
└__call__
+┴&call_and_return_all_conditional_losses
	┬axis
	gamma
 beta*
м
├	variables
─trainable_variables
┼regularization_losses
╞	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses
╔_random_generator* 
м
╩	variables
╦trainable_variables
╠regularization_losses
═	keras_api
╬__call__
+╧&call_and_return_all_conditional_losses
╨_random_generator* 

!0
"1*

!0
"1*
* 
Ш
╤non_trainable_variables
╥layers
╙metrics
 ╘layer_regularization_losses
╒layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

╓trace_0* 

╫trace_0* 
* 

#0
$1*

#0
$1*
* 
Ш
╪non_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

▌trace_0* 

▐trace_0* 
* 
* 
* 
Ц
▀non_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 

фtrace_0
хtrace_1* 

цtrace_0
чtrace_1* 
* 

%0
&1*

%0
&1*
* 
Ы
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*

эtrace_0* 

юtrace_0* 
* 
 
A0
B1
C2
D3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

С0
Т1*

П	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Х0
Ц1*

У	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
.
b0
c1
d2
e3
f4
g5*
* 
* 
* 
* 
* 
* 
* 
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
Ю
яnon_trainable_variables
Ёlayers
ёmetrics
 Єlayer_regularization_losses
єlayer_metrics
б	variables
вtrainable_variables
гregularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses*
* 
* 
м
Ї	variables
їtrainable_variables
Ўregularization_losses
ў	keras_api
°__call__
+∙&call_and_return_all_conditional_losses

kernel
bias*
м
·	variables
√trainable_variables
№regularization_losses
¤	keras_api
■__call__
+ &call_and_return_all_conditional_losses

kernel
bias*
м
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses

kernel
bias*
м
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses

kernel
bias*
м
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses

kernel
bias*
м
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses
Ш_random_generator* 
м
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses

kernel
bias*
м
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses
е_random_generator* 
 
0
1
2
3*
 
0
1
2
3*
* 
Ю
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
п	variables
░trainable_variables
▒regularization_losses
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses*
:
лtrace_0
мtrace_1
нtrace_2
оtrace_3* 
:
пtrace_0
░trace_1
▒trace_2
▓trace_3* 

0
1*

0
1*
* 
Ю
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses*
* 
* 
* 

0
 1*

0
 1*
* 
Ю
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
╝	variables
╜trainable_variables
╛regularization_losses
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
Ь
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
├	variables
─trainable_variables
┼regularization_losses
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
Ь
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
╞layer_metrics
╩	variables
╦trainable_variables
╠regularization_losses
╬__call__
+╧&call_and_return_all_conditional_losses
'╧"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
$
з0
и1
й2
к3*
* 
* 
* 

0
1*

0
1*
* 
Ю
╟non_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
Ї	variables
їtrainable_variables
Ўregularization_losses
°__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
Ю
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
·	variables
√trainable_variables
№regularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
Ю
╤non_trainable_variables
╥layers
╙metrics
 ╘layer_regularization_losses
╒layer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
Ю
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 
Ю
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses*

рtrace_0* 

сtrace_0* 
* 
* 
* 
Ь
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses* 

чtrace_0
шtrace_1* 

щtrace_0
ъtrace_1* 
* 

0
1*

0
1*
* 
Ю
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses*

Ёtrace_0* 

ёtrace_0* 
* 
* 
* 
Ь
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses* 

ўtrace_0
°trace_1* 

∙trace_0
·trace_1* 
* 
* 
$
л0
м1
н2
о3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ь
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamepos_emb/Read/ReadVariableOpclass_emb/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/query/kernel/Read/ReadVariableOpJtransformer_block/multi_head_self_attention/query/bias/Read/ReadVariableOpJtransformer_block/multi_head_self_attention/key/kernel/Read/ReadVariableOpHtransformer_block/multi_head_self_attention/key/bias/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/value/kernel/Read/ReadVariableOpJtransformer_block/multi_head_self_attention/value/bias/Read/ReadVariableOpJtransformer_block/multi_head_self_attention/out/kernel/Read/ReadVariableOpHtransformer_block/multi_head_self_attention/out/bias/Read/ReadVariableOp$dense_mlp/kernel/Read/ReadVariableOp"dense_mlp/bias/Read/ReadVariableOp*dense_embed_dim/kernel/Read/ReadVariableOp(dense_embed_dim/bias/Read/ReadVariableOp?transformer_block/layer_normalization/gamma/Read/ReadVariableOp>transformer_block/layer_normalization/beta/Read/ReadVariableOpAtransformer_block/layer_normalization_1/gamma/Read/ReadVariableOp@transformer_block/layer_normalization_1/beta/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_38345
┐	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamepos_emb	class_embdense/kernel
dense/bias8transformer_block/multi_head_self_attention/query/kernel6transformer_block/multi_head_self_attention/query/bias6transformer_block/multi_head_self_attention/key/kernel4transformer_block/multi_head_self_attention/key/bias8transformer_block/multi_head_self_attention/value/kernel6transformer_block/multi_head_self_attention/value/bias6transformer_block/multi_head_self_attention/out/kernel4transformer_block/multi_head_self_attention/out/biasdense_mlp/kerneldense_mlp/biasdense_embed_dim/kerneldense_embed_dim/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betalayer_normalization_2/gammalayer_normalization_2/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biastotal_1count_1totalcount**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_38445й╔&
я
Б
,__inference_sequential_1_layer_call_fn_37098

inputs
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 

	unknown_4:

identityИвStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_34698o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
и╗
Ч
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36642
x9
'dense_tensordot_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@9
#broadcastto_readvariableop_resource:@1
add_readvariableop_resource:A@Y
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:@U
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource:@e
Stransformer_block_multi_head_self_attention_query_tensordot_readvariableop_resource:@@_
Qtransformer_block_multi_head_self_attention_query_biasadd_readvariableop_resource:@c
Qtransformer_block_multi_head_self_attention_key_tensordot_readvariableop_resource:@@]
Otransformer_block_multi_head_self_attention_key_biasadd_readvariableop_resource:@e
Stransformer_block_multi_head_self_attention_value_tensordot_readvariableop_resource:@@_
Qtransformer_block_multi_head_self_attention_value_biasadd_readvariableop_resource:@c
Qtransformer_block_multi_head_self_attention_out_tensordot_readvariableop_resource:@@]
Otransformer_block_multi_head_self_attention_out_biasadd_readvariableop_resource:@[
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:@W
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource:@Z
Htransformer_block_sequential_dense_mlp_tensordot_readvariableop_resource:@ T
Ftransformer_block_sequential_dense_mlp_biasadd_readvariableop_resource: `
Ntransformer_block_sequential_dense_embed_dim_tensordot_readvariableop_resource: @Z
Ltransformer_block_sequential_dense_embed_dim_biasadd_readvariableop_resource:@V
Hsequential_1_layer_normalization_2_batchnorm_mul_readvariableop_resource:@R
Dsequential_1_layer_normalization_2_batchnorm_readvariableop_resource:@E
3sequential_1_dense_1_matmul_readvariableop_resource:@ B
4sequential_1_dense_1_biasadd_readvariableop_resource: E
3sequential_1_dense_2_matmul_readvariableop_resource: 
B
4sequential_1_dense_2_biasadd_readvariableop_resource:

identityИвBroadcastTo/ReadVariableOpвadd/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpв+sequential_1/dense_1/BiasAdd/ReadVariableOpв*sequential_1/dense_1/MatMul/ReadVariableOpв+sequential_1/dense_2/BiasAdd/ReadVariableOpв*sequential_1/dense_2/MatMul/ReadVariableOpв;sequential_1/layer_normalization_2/batchnorm/ReadVariableOpв?sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOpв>transformer_block/layer_normalization/batchnorm/ReadVariableOpвBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpв@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpвDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpвFtransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpвHtransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpвFtransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpвHtransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpвHtransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpвJtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpвHtransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpвJtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpвCtransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpвEtransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpв=transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOpв?transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp6
ShapeShapex*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;W
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    l
rescaling/mulMulxrescaling/Cast/x:output:0*
T0*/
_output_shapes
:           А
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*/
_output_shapes
:           H
Shape_1Shaperescaling/add:z:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╟
ExtractImagePatchesExtractImagePatchesrescaling/add:z:0*
T0*/
_output_shapes
:         *
ksizes
*
paddingVALID*
rates
*
strides
Z
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :С
Reshape/shapePackstrided_slice_1:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:И
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*4
_output_shapes"
 :                  Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       U
dense/Tensordot/ShapeShapeReshape:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
dense/Tensordot/transpose	TransposeReshape:output:0dense/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ю
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @В
BroadcastTo/ReadVariableOpReadVariableOp#broadcastto_readvariableop_resource*"
_output_shapes
:@*
dtype0U
BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :U
BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@Ы
BroadcastTo/shapePackstrided_slice:output:0BroadcastTo/shape/1:output:0BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:Р
BroadcastToBroadcastTo"BroadcastTo/ReadVariableOp:value:0BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         @M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ю
concatConcatV2BroadcastTo:output:0dense/BiasAdd:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  @r
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*"
_output_shapes
:A@*
dtype0o
addAddV2concat:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@О
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┘
2transformer_block/layer_normalization/moments/meanMeanadd:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(╜
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         A╪
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@Т
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Э
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(z
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5є
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         Aн
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         A╩
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ў
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@┤
5transformer_block/layer_normalization/batchnorm/mul_1Muladd:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@ш
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@┬
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0є
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@ш
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@Ъ
1transformer_block/multi_head_self_attention/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Atransformer_block/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Atransformer_block/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
9transformer_block/multi_head_self_attention/strided_sliceStridedSlice:transformer_block/multi_head_self_attention/Shape:output:0Htransformer_block/multi_head_self_attention/strided_slice/stack:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask▐
Jtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_query_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0К
@transformer_block/multi_head_self_attention/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:С
@transformer_block/multi_head_self_attention/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       к
Atransformer_block/multi_head_self_attention/query/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Л
Itransformer_block/multi_head_self_attention/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
Dtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2GatherV2Jtransformer_block/multi_head_self_attention/query/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/query/Tensordot/free:output:0Rtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Н
Ktransformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
Ftransformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1GatherV2Jtransformer_block/multi_head_self_attention/query/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/query/Tensordot/axes:output:0Ttransformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Л
Atransformer_block/multi_head_self_attention/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Д
@transformer_block/multi_head_self_attention/query/Tensordot/ProdProdMtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/query/Tensordot/Const:output:0*
T0*
_output_shapes
: Н
Ctransformer_block/multi_head_self_attention/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: К
Btransformer_block/multi_head_self_attention/query/Tensordot/Prod_1ProdOtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1:output:0Ltransformer_block/multi_head_self_attention/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Й
Gtransformer_block/multi_head_self_attention/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
Btransformer_block/multi_head_self_attention/query/Tensordot/concatConcatV2Itransformer_block/multi_head_self_attention/query/Tensordot/free:output:0Itransformer_block/multi_head_self_attention/query/Tensordot/axes:output:0Ptransformer_block/multi_head_self_attention/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:П
Atransformer_block/multi_head_self_attention/query/Tensordot/stackPackItransformer_block/multi_head_self_attention/query/Tensordot/Prod:output:0Ktransformer_block/multi_head_self_attention/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Р
Etransformer_block/multi_head_self_attention/query/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0Ktransformer_block/multi_head_self_attention/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@а
Ctransformer_block/multi_head_self_attention/query/Tensordot/ReshapeReshapeItransformer_block/multi_head_self_attention/query/Tensordot/transpose:y:0Jtransformer_block/multi_head_self_attention/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  а
Btransformer_block/multi_head_self_attention/query/Tensordot/MatMulMatMulLtransformer_block/multi_head_self_attention/query/Tensordot/Reshape:output:0Rtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Н
Ctransformer_block/multi_head_self_attention/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Л
Itransformer_block/multi_head_self_attention/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
Dtransformer_block/multi_head_self_attention/query/Tensordot/concat_1ConcatV2Mtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/query/Tensordot/Const_2:output:0Rtransformer_block/multi_head_self_attention/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Щ
;transformer_block/multi_head_self_attention/query/TensordotReshapeLtransformer_block/multi_head_self_attention/query/Tensordot/MatMul:product:0Mtransformer_block/multi_head_self_attention/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@╓
Htransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_query_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Т
9transformer_block/multi_head_self_attention/query/BiasAddBiasAddDtransformer_block/multi_head_self_attention/query/Tensordot:output:0Ptransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@┌
Htransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_key_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0И
>transformer_block/multi_head_self_attention/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:П
>transformer_block/multi_head_self_attention/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       и
?transformer_block/multi_head_self_attention/key/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Й
Gtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
Btransformer_block/multi_head_self_attention/key/Tensordot/GatherV2GatherV2Htransformer_block/multi_head_self_attention/key/Tensordot/Shape:output:0Gtransformer_block/multi_head_self_attention/key/Tensordot/free:output:0Ptransformer_block/multi_head_self_attention/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Л
Itransformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
Dtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1GatherV2Htransformer_block/multi_head_self_attention/key/Tensordot/Shape:output:0Gtransformer_block/multi_head_self_attention/key/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ■
>transformer_block/multi_head_self_attention/key/Tensordot/ProdProdKtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2:output:0Htransformer_block/multi_head_self_attention/key/Tensordot/Const:output:0*
T0*
_output_shapes
: Л
Atransformer_block/multi_head_self_attention/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Д
@transformer_block/multi_head_self_attention/key/Tensordot/Prod_1ProdMtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1:output:0Jtransformer_block/multi_head_self_attention/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: З
Etransformer_block/multi_head_self_attention/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
@transformer_block/multi_head_self_attention/key/Tensordot/concatConcatV2Gtransformer_block/multi_head_self_attention/key/Tensordot/free:output:0Gtransformer_block/multi_head_self_attention/key/Tensordot/axes:output:0Ntransformer_block/multi_head_self_attention/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/key/Tensordot/stackPackGtransformer_block/multi_head_self_attention/key/Tensordot/Prod:output:0Itransformer_block/multi_head_self_attention/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:М
Ctransformer_block/multi_head_self_attention/key/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0Itransformer_block/multi_head_self_attention/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@Ъ
Atransformer_block/multi_head_self_attention/key/Tensordot/ReshapeReshapeGtransformer_block/multi_head_self_attention/key/Tensordot/transpose:y:0Htransformer_block/multi_head_self_attention/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ъ
@transformer_block/multi_head_self_attention/key/Tensordot/MatMulMatMulJtransformer_block/multi_head_self_attention/key/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Л
Atransformer_block/multi_head_self_attention/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Й
Gtransformer_block/multi_head_self_attention/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Btransformer_block/multi_head_self_attention/key/Tensordot/concat_1ConcatV2Ktransformer_block/multi_head_self_attention/key/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/key/Tensordot/Const_2:output:0Ptransformer_block/multi_head_self_attention/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:У
9transformer_block/multi_head_self_attention/key/TensordotReshapeJtransformer_block/multi_head_self_attention/key/Tensordot/MatMul:product:0Ktransformer_block/multi_head_self_attention/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@╥
Ftransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpReadVariableOpOtransformer_block_multi_head_self_attention_key_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
7transformer_block/multi_head_self_attention/key/BiasAddBiasAddBtransformer_block/multi_head_self_attention/key/Tensordot:output:0Ntransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@▐
Jtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_value_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0К
@transformer_block/multi_head_self_attention/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:С
@transformer_block/multi_head_self_attention/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       к
Atransformer_block/multi_head_self_attention/value/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Л
Itransformer_block/multi_head_self_attention/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
Dtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2GatherV2Jtransformer_block/multi_head_self_attention/value/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/value/Tensordot/free:output:0Rtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Н
Ktransformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
Ftransformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1GatherV2Jtransformer_block/multi_head_self_attention/value/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/value/Tensordot/axes:output:0Ttransformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Л
Atransformer_block/multi_head_self_attention/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Д
@transformer_block/multi_head_self_attention/value/Tensordot/ProdProdMtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/value/Tensordot/Const:output:0*
T0*
_output_shapes
: Н
Ctransformer_block/multi_head_self_attention/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: К
Btransformer_block/multi_head_self_attention/value/Tensordot/Prod_1ProdOtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1:output:0Ltransformer_block/multi_head_self_attention/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Й
Gtransformer_block/multi_head_self_attention/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
Btransformer_block/multi_head_self_attention/value/Tensordot/concatConcatV2Itransformer_block/multi_head_self_attention/value/Tensordot/free:output:0Itransformer_block/multi_head_self_attention/value/Tensordot/axes:output:0Ptransformer_block/multi_head_self_attention/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:П
Atransformer_block/multi_head_self_attention/value/Tensordot/stackPackItransformer_block/multi_head_self_attention/value/Tensordot/Prod:output:0Ktransformer_block/multi_head_self_attention/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Р
Etransformer_block/multi_head_self_attention/value/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0Ktransformer_block/multi_head_self_attention/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@а
Ctransformer_block/multi_head_self_attention/value/Tensordot/ReshapeReshapeItransformer_block/multi_head_self_attention/value/Tensordot/transpose:y:0Jtransformer_block/multi_head_self_attention/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  а
Btransformer_block/multi_head_self_attention/value/Tensordot/MatMulMatMulLtransformer_block/multi_head_self_attention/value/Tensordot/Reshape:output:0Rtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Н
Ctransformer_block/multi_head_self_attention/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Л
Itransformer_block/multi_head_self_attention/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
Dtransformer_block/multi_head_self_attention/value/Tensordot/concat_1ConcatV2Mtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/value/Tensordot/Const_2:output:0Rtransformer_block/multi_head_self_attention/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Щ
;transformer_block/multi_head_self_attention/value/TensordotReshapeLtransformer_block/multi_head_self_attention/value/Tensordot/MatMul:product:0Mtransformer_block/multi_head_self_attention/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@╓
Htransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_value_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Т
9transformer_block/multi_head_self_attention/value/BiasAddBiasAddDtransformer_block/multi_head_self_attention/value/Tensordot:output:0Ptransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@Ж
;transformer_block/multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         }
;transformer_block/multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}
;transformer_block/multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
9transformer_block/multi_head_self_attention/Reshape/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/1:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/2:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Й
3transformer_block/multi_head_self_attention/ReshapeReshapeBtransformer_block/multi_head_self_attention/query/BiasAdd:output:0Btransformer_block/multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"                  У
:transformer_block/multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             И
5transformer_block/multi_head_self_attention/transpose	Transpose<transformer_block/multi_head_self_attention/Reshape:output:0Ctransformer_block/multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"                  И
=transformer_block/multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         
=transformer_block/multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
=transformer_block/multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
;transformer_block/multi_head_self_attention/Reshape_1/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:Л
5transformer_block/multi_head_self_attention/Reshape_1Reshape@transformer_block/multi_head_self_attention/key/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"                  Х
<transformer_block/multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             О
7transformer_block/multi_head_self_attention/transpose_1	Transpose>transformer_block/multi_head_self_attention/Reshape_1:output:0Etransformer_block/multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"                  И
=transformer_block/multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         
=transformer_block/multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
=transformer_block/multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
;transformer_block/multi_head_self_attention/Reshape_2/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:Н
5transformer_block/multi_head_self_attention/Reshape_2ReshapeBtransformer_block/multi_head_self_attention/value/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"                  Х
<transformer_block/multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             О
7transformer_block/multi_head_self_attention/transpose_2	Transpose>transformer_block/multi_head_self_attention/Reshape_2:output:0Etransformer_block/multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"                  Ф
2transformer_block/multi_head_self_attention/MatMulBatchMatMulV29transformer_block/multi_head_self_attention/transpose:y:0;transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+                           *
adj_y(Ю
3transformer_block/multi_head_self_attention/Shape_1Shape;transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:Ф
Atransformer_block/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         Н
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Н
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╖
;transformer_block/multi_head_self_attention/strided_slice_1StridedSlice<transformer_block/multi_head_self_attention/Shape_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice_1/stack:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_1:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskо
0transformer_block/multi_head_self_attention/CastCastDtransformer_block/multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: П
0transformer_block/multi_head_self_attention/SqrtSqrt4transformer_block/multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: ¤
3transformer_block/multi_head_self_attention/truedivRealDiv;transformer_block/multi_head_self_attention/MatMul:output:04transformer_block/multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+                           ├
3transformer_block/multi_head_self_attention/SoftmaxSoftmax7transformer_block/multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+                           Д
4transformer_block/multi_head_self_attention/MatMul_1BatchMatMulV2=transformer_block/multi_head_self_attention/Softmax:softmax:0;transformer_block/multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"                  Х
<transformer_block/multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Н
7transformer_block/multi_head_self_attention/transpose_3	Transpose=transformer_block/multi_head_self_attention/MatMul_1:output:0Etransformer_block/multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"                  И
=transformer_block/multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         
=transformer_block/multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@┼
;transformer_block/multi_head_self_attention/Reshape_3/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:В
5transformer_block/multi_head_self_attention/Reshape_3Reshape;transformer_block/multi_head_self_attention/transpose_3:y:0Dtransformer_block/multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :                  @┌
Htransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_out_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0И
>transformer_block/multi_head_self_attention/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:П
>transformer_block/multi_head_self_attention/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       н
?transformer_block/multi_head_self_attention/out/Tensordot/ShapeShape>transformer_block/multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:Й
Gtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
Btransformer_block/multi_head_self_attention/out/Tensordot/GatherV2GatherV2Htransformer_block/multi_head_self_attention/out/Tensordot/Shape:output:0Gtransformer_block/multi_head_self_attention/out/Tensordot/free:output:0Ptransformer_block/multi_head_self_attention/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Л
Itransformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
Dtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1GatherV2Htransformer_block/multi_head_self_attention/out/Tensordot/Shape:output:0Gtransformer_block/multi_head_self_attention/out/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ■
>transformer_block/multi_head_self_attention/out/Tensordot/ProdProdKtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2:output:0Htransformer_block/multi_head_self_attention/out/Tensordot/Const:output:0*
T0*
_output_shapes
: Л
Atransformer_block/multi_head_self_attention/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Д
@transformer_block/multi_head_self_attention/out/Tensordot/Prod_1ProdMtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1:output:0Jtransformer_block/multi_head_self_attention/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: З
Etransformer_block/multi_head_self_attention/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
@transformer_block/multi_head_self_attention/out/Tensordot/concatConcatV2Gtransformer_block/multi_head_self_attention/out/Tensordot/free:output:0Gtransformer_block/multi_head_self_attention/out/Tensordot/axes:output:0Ntransformer_block/multi_head_self_attention/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/out/Tensordot/stackPackGtransformer_block/multi_head_self_attention/out/Tensordot/Prod:output:0Itransformer_block/multi_head_self_attention/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ъ
Ctransformer_block/multi_head_self_attention/out/Tensordot/transpose	Transpose>transformer_block/multi_head_self_attention/Reshape_3:output:0Itransformer_block/multi_head_self_attention/out/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @Ъ
Atransformer_block/multi_head_self_attention/out/Tensordot/ReshapeReshapeGtransformer_block/multi_head_self_attention/out/Tensordot/transpose:y:0Htransformer_block/multi_head_self_attention/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ъ
@transformer_block/multi_head_self_attention/out/Tensordot/MatMulMatMulJtransformer_block/multi_head_self_attention/out/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Л
Atransformer_block/multi_head_self_attention/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Й
Gtransformer_block/multi_head_self_attention/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Btransformer_block/multi_head_self_attention/out/Tensordot/concat_1ConcatV2Ktransformer_block/multi_head_self_attention/out/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/out/Tensordot/Const_2:output:0Ptransformer_block/multi_head_self_attention/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ь
9transformer_block/multi_head_self_attention/out/TensordotReshapeJtransformer_block/multi_head_self_attention/out/Tensordot/MatMul:product:0Ktransformer_block/multi_head_self_attention/out/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @╥
Ftransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpReadVariableOpOtransformer_block_multi_head_self_attention_out_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Х
7transformer_block/multi_head_self_attention/out/BiasAddBiasAddBtransformer_block/multi_head_self_attention/out/Tensordot:output:0Ntransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @▒
$transformer_block/dropout_2/IdentityIdentity@transformer_block/multi_head_self_attention/out/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @М
transformer_block/addAddV2-transformer_block/dropout_2/Identity:output:0add:z:0*
T0*+
_output_shapes
:         A@Р
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:я
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(┴
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         Aю
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@Ф
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:г
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(|
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5∙
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         A▒
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         A╬
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0¤
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@╩
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@ю
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@╞
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0∙
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@ю
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@╚
?transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOpReadVariableOpHtransformer_block_sequential_dense_mlp_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0
5transformer_block/sequential/dense_mlp/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Ж
5transformer_block/sequential/dense_mlp/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       б
6transformer_block/sequential/dense_mlp/Tensordot/ShapeShape;transformer_block/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:А
>transformer_block/sequential/dense_mlp/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
9transformer_block/sequential/dense_mlp/Tensordot/GatherV2GatherV2?transformer_block/sequential/dense_mlp/Tensordot/Shape:output:0>transformer_block/sequential/dense_mlp/Tensordot/free:output:0Gtransformer_block/sequential/dense_mlp/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:В
@transformer_block/sequential/dense_mlp/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : █
;transformer_block/sequential/dense_mlp/Tensordot/GatherV2_1GatherV2?transformer_block/sequential/dense_mlp/Tensordot/Shape:output:0>transformer_block/sequential/dense_mlp/Tensordot/axes:output:0Itransformer_block/sequential/dense_mlp/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:А
6transformer_block/sequential/dense_mlp/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: у
5transformer_block/sequential/dense_mlp/Tensordot/ProdProdBtransformer_block/sequential/dense_mlp/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_mlp/Tensordot/Const:output:0*
T0*
_output_shapes
: В
8transformer_block/sequential/dense_mlp/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: щ
7transformer_block/sequential/dense_mlp/Tensordot/Prod_1ProdDtransformer_block/sequential/dense_mlp/Tensordot/GatherV2_1:output:0Atransformer_block/sequential/dense_mlp/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_mlp/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╕
7transformer_block/sequential/dense_mlp/Tensordot/concatConcatV2>transformer_block/sequential/dense_mlp/Tensordot/free:output:0>transformer_block/sequential/dense_mlp/Tensordot/axes:output:0Etransformer_block/sequential/dense_mlp/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ю
6transformer_block/sequential/dense_mlp/Tensordot/stackPack>transformer_block/sequential/dense_mlp/Tensordot/Prod:output:0@transformer_block/sequential/dense_mlp/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:№
:transformer_block/sequential/dense_mlp/Tensordot/transpose	Transpose;transformer_block/layer_normalization_1/batchnorm/add_1:z:0@transformer_block/sequential/dense_mlp/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ 
8transformer_block/sequential/dense_mlp/Tensordot/ReshapeReshape>transformer_block/sequential/dense_mlp/Tensordot/transpose:y:0?transformer_block/sequential/dense_mlp/Tensordot/stack:output:0*
T0*0
_output_shapes
:                   
7transformer_block/sequential/dense_mlp/Tensordot/MatMulMatMulAtransformer_block/sequential/dense_mlp/Tensordot/Reshape:output:0Gtransformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          В
8transformer_block/sequential/dense_mlp/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: А
>transformer_block/sequential/dense_mlp/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ├
9transformer_block/sequential/dense_mlp/Tensordot/concat_1ConcatV2Btransformer_block/sequential/dense_mlp/Tensordot/GatherV2:output:0Atransformer_block/sequential/dense_mlp/Tensordot/Const_2:output:0Gtransformer_block/sequential/dense_mlp/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:°
0transformer_block/sequential/dense_mlp/TensordotReshapeAtransformer_block/sequential/dense_mlp/Tensordot/MatMul:product:0Btransformer_block/sequential/dense_mlp/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A └
=transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_mlp_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ё
.transformer_block/sequential/dense_mlp/BiasAddBiasAdd9transformer_block/sequential/dense_mlp/Tensordot:output:0Etransformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A v
1transformer_block/sequential/dense_mlp/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?с
/transformer_block/sequential/dense_mlp/Gelu/mulMul:transformer_block/sequential/dense_mlp/Gelu/mul/x:output:07transformer_block/sequential/dense_mlp/BiasAdd:output:0*
T0*+
_output_shapes
:         A w
2transformer_block/sequential/dense_mlp/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?ъ
3transformer_block/sequential/dense_mlp/Gelu/truedivRealDiv7transformer_block/sequential/dense_mlp/BiasAdd:output:0;transformer_block/sequential/dense_mlp/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A е
/transformer_block/sequential/dense_mlp/Gelu/ErfErf7transformer_block/sequential/dense_mlp/Gelu/truediv:z:0*
T0*+
_output_shapes
:         A v
1transformer_block/sequential/dense_mlp/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?▀
/transformer_block/sequential/dense_mlp/Gelu/addAddV2:transformer_block/sequential/dense_mlp/Gelu/add/x:output:03transformer_block/sequential/dense_mlp/Gelu/Erf:y:0*
T0*+
_output_shapes
:         A ╪
1transformer_block/sequential/dense_mlp/Gelu/mul_1Mul3transformer_block/sequential/dense_mlp/Gelu/mul:z:03transformer_block/sequential/dense_mlp/Gelu/add:z:0*
T0*+
_output_shapes
:         A ж
-transformer_block/sequential/dropout/IdentityIdentity5transformer_block/sequential/dense_mlp/Gelu/mul_1:z:0*
T0*+
_output_shapes
:         A ╘
Etransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpReadVariableOpNtransformer_block_sequential_dense_embed_dim_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0Е
;transformer_block/sequential/dense_embed_dim/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:М
;transformer_block/sequential/dense_embed_dim/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       в
<transformer_block/sequential/dense_embed_dim/Tensordot/ShapeShape6transformer_block/sequential/dropout/Identity:output:0*
T0*
_output_shapes
:Ж
Dtransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : я
?transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2GatherV2Etransformer_block/sequential/dense_embed_dim/Tensordot/Shape:output:0Dtransformer_block/sequential/dense_embed_dim/Tensordot/free:output:0Mtransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:И
Ftransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : є
Atransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1GatherV2Etransformer_block/sequential/dense_embed_dim/Tensordot/Shape:output:0Dtransformer_block/sequential/dense_embed_dim/Tensordot/axes:output:0Otransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ж
<transformer_block/sequential/dense_embed_dim/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ї
;transformer_block/sequential/dense_embed_dim/Tensordot/ProdProdHtransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2:output:0Etransformer_block/sequential/dense_embed_dim/Tensordot/Const:output:0*
T0*
_output_shapes
: И
>transformer_block/sequential/dense_embed_dim/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: √
=transformer_block/sequential/dense_embed_dim/Tensordot/Prod_1ProdJtransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1:output:0Gtransformer_block/sequential/dense_embed_dim/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Д
Btransformer_block/sequential/dense_embed_dim/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╨
=transformer_block/sequential/dense_embed_dim/Tensordot/concatConcatV2Dtransformer_block/sequential/dense_embed_dim/Tensordot/free:output:0Dtransformer_block/sequential/dense_embed_dim/Tensordot/axes:output:0Ktransformer_block/sequential/dense_embed_dim/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:А
<transformer_block/sequential/dense_embed_dim/Tensordot/stackPackDtransformer_block/sequential/dense_embed_dim/Tensordot/Prod:output:0Ftransformer_block/sequential/dense_embed_dim/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Г
@transformer_block/sequential/dense_embed_dim/Tensordot/transpose	Transpose6transformer_block/sequential/dropout/Identity:output:0Ftransformer_block/sequential/dense_embed_dim/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A С
>transformer_block/sequential/dense_embed_dim/Tensordot/ReshapeReshapeDtransformer_block/sequential/dense_embed_dim/Tensordot/transpose:y:0Etransformer_block/sequential/dense_embed_dim/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  С
=transformer_block/sequential/dense_embed_dim/Tensordot/MatMulMatMulGtransformer_block/sequential/dense_embed_dim/Tensordot/Reshape:output:0Mtransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @И
>transformer_block/sequential/dense_embed_dim/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Ж
Dtransformer_block/sequential/dense_embed_dim/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : █
?transformer_block/sequential/dense_embed_dim/Tensordot/concat_1ConcatV2Htransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2:output:0Gtransformer_block/sequential/dense_embed_dim/Tensordot/Const_2:output:0Mtransformer_block/sequential/dense_embed_dim/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:К
6transformer_block/sequential/dense_embed_dim/TensordotReshapeGtransformer_block/sequential/dense_embed_dim/Tensordot/MatMul:product:0Htransformer_block/sequential/dense_embed_dim/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@╠
Ctransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpReadVariableOpLtransformer_block_sequential_dense_embed_dim_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
4transformer_block/sequential/dense_embed_dim/BiasAddBiasAdd?transformer_block/sequential/dense_embed_dim/Tensordot:output:0Ktransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@░
/transformer_block/sequential/dropout_1/IdentityIdentity=transformer_block/sequential/dense_embed_dim/BiasAdd:output:0*
T0*+
_output_shapes
:         A@а
$transformer_block/dropout_3/IdentityIdentity8transformer_block/sequential/dropout_1/Identity:output:0*
T0*+
_output_shapes
:         A@а
transformer_block/add_1AddV2-transformer_block/dropout_3/Identity:output:0transformer_block/add:z:0*
T0*+
_output_shapes
:         A@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Щ
strided_slice_2StridedSlicetransformer_block/add_1:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_maskЛ
Asequential_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:р
/sequential_1/layer_normalization_2/moments/meanMeanstrided_slice_2:output:0Jsequential_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(│
7sequential_1/layer_normalization_2/moments/StopGradientStopGradient8sequential_1/layer_normalization_2/moments/mean:output:0*
T0*'
_output_shapes
:         ▀
<sequential_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencestrided_slice_2:output:0@sequential_1/layer_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @П
Esequential_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Р
3sequential_1/layer_normalization_2/moments/varianceMean@sequential_1/layer_normalization_2/moments/SquaredDifference:z:0Nsequential_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(w
2sequential_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5ц
0sequential_1/layer_normalization_2/batchnorm/addAddV2<sequential_1/layer_normalization_2/moments/variance:output:0;sequential_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         г
2sequential_1/layer_normalization_2/batchnorm/RsqrtRsqrt4sequential_1/layer_normalization_2/batchnorm/add:z:0*
T0*'
_output_shapes
:         ─
?sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ъ
0sequential_1/layer_normalization_2/batchnorm/mulMul6sequential_1/layer_normalization_2/batchnorm/Rsqrt:y:0Gsequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╗
2sequential_1/layer_normalization_2/batchnorm/mul_1Mulstrided_slice_2:output:04sequential_1/layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @█
2sequential_1/layer_normalization_2/batchnorm/mul_2Mul8sequential_1/layer_normalization_2/moments/mean:output:04sequential_1/layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @╝
;sequential_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpDsequential_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0ц
0sequential_1/layer_normalization_2/batchnorm/subSubCsequential_1/layer_normalization_2/batchnorm/ReadVariableOp:value:06sequential_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*'
_output_shapes
:         @█
2sequential_1/layer_normalization_2/batchnorm/add_1AddV26sequential_1/layer_normalization_2/batchnorm/mul_1:z:04sequential_1/layer_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @Ю
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0├
sequential_1/dense_1/MatMulMatMul6sequential_1/layer_normalization_2/batchnorm/add_1:z:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Ь
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╡
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
sequential_1/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
sequential_1/dense_1/Gelu/mulMul(sequential_1/dense_1/Gelu/mul/x:output:0%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          e
 sequential_1/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?░
!sequential_1/dense_1/Gelu/truedivRealDiv%sequential_1/dense_1/BiasAdd:output:0)sequential_1/dense_1/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:          }
sequential_1/dense_1/Gelu/ErfErf%sequential_1/dense_1/Gelu/truediv:z:0*
T0*'
_output_shapes
:          d
sequential_1/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?е
sequential_1/dense_1/Gelu/addAddV2(sequential_1/dense_1/Gelu/add/x:output:0!sequential_1/dense_1/Gelu/Erf:y:0*
T0*'
_output_shapes
:          Ю
sequential_1/dense_1/Gelu/mul_1Mul!sequential_1/dense_1/Gelu/mul:z:0!sequential_1/dense_1/Gelu/add:z:0*
T0*'
_output_shapes
:          В
sequential_1/dropout_4/IdentityIdentity#sequential_1/dense_1/Gelu/mul_1:z:0*
T0*'
_output_shapes
:          Ю
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0╡
sequential_1/dense_2/MatMulMatMul(sequential_1/dropout_4/Identity:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Ь
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0╡
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
t
IdentityIdentity%sequential_1/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
ц
NoOpNoOp^BroadcastTo/ReadVariableOp^add/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp<^sequential_1/layer_normalization_2/batchnorm/ReadVariableOp@^sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpG^transformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpI^transformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpG^transformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpI^transformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpI^transformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpK^transformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpI^transformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpK^transformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpD^transformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpF^transformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOp>^transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp@^transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 28
BroadcastTo/ReadVariableOpBroadcastTo/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2z
;sequential_1/layer_normalization_2/batchnorm/ReadVariableOp;sequential_1/layer_normalization_2/batchnorm/ReadVariableOp2В
?sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2А
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2И
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2Д
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2М
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2Р
Ftransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpFtransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOp2Ф
Htransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpHtransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOp2Р
Ftransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpFtransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOp2Ф
Htransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpHtransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOp2Ф
Htransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpHtransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOp2Ш
Jtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpJtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOp2Ф
Htransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpHtransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOp2Ш
Jtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpJtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOp2К
Ctransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpCtransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOp2О
Etransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpEtransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOp2~
=transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp=transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp2В
?transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp?transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp:R N
/
_output_shapes
:           

_user_specified_namex
▓
є
B__inference_dense_1_layer_call_and_return_conditional_losses_37874

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?h
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:          P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?q
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:          S
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:          O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:          _

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:          ]
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Е[
№
E__inference_sequential_layer_call_and_return_conditional_losses_38092

inputs=
+dense_mlp_tensordot_readvariableop_resource:@ 7
)dense_mlp_biasadd_readvariableop_resource: C
1dense_embed_dim_tensordot_readvariableop_resource: @=
/dense_embed_dim_biasadd_readvariableop_resource:@
identityИв&dense_embed_dim/BiasAdd/ReadVariableOpв(dense_embed_dim/Tensordot/ReadVariableOpв dense_mlp/BiasAdd/ReadVariableOpв"dense_mlp/Tensordot/ReadVariableOpО
"dense_mlp/Tensordot/ReadVariableOpReadVariableOp+dense_mlp_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0b
dense_mlp/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense_mlp/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       O
dense_mlp/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:c
!dense_mlp/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
dense_mlp/Tensordot/GatherV2GatherV2"dense_mlp/Tensordot/Shape:output:0!dense_mlp/Tensordot/free:output:0*dense_mlp/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#dense_mlp/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
dense_mlp/Tensordot/GatherV2_1GatherV2"dense_mlp/Tensordot/Shape:output:0!dense_mlp/Tensordot/axes:output:0,dense_mlp/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
dense_mlp/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
dense_mlp/Tensordot/ProdProd%dense_mlp/Tensordot/GatherV2:output:0"dense_mlp/Tensordot/Const:output:0*
T0*
_output_shapes
: e
dense_mlp/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
dense_mlp/Tensordot/Prod_1Prod'dense_mlp/Tensordot/GatherV2_1:output:0$dense_mlp/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
dense_mlp/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
dense_mlp/Tensordot/concatConcatV2!dense_mlp/Tensordot/free:output:0!dense_mlp/Tensordot/axes:output:0(dense_mlp/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
dense_mlp/Tensordot/stackPack!dense_mlp/Tensordot/Prod:output:0#dense_mlp/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Н
dense_mlp/Tensordot/transpose	Transposeinputs#dense_mlp/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@и
dense_mlp/Tensordot/ReshapeReshape!dense_mlp/Tensordot/transpose:y:0"dense_mlp/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
dense_mlp/Tensordot/MatMulMatMul$dense_mlp/Tensordot/Reshape:output:0*dense_mlp/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          e
dense_mlp/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: c
!dense_mlp/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
dense_mlp/Tensordot/concat_1ConcatV2%dense_mlp/Tensordot/GatherV2:output:0$dense_mlp/Tensordot/Const_2:output:0*dense_mlp/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:б
dense_mlp/TensordotReshape$dense_mlp/Tensordot/MatMul:product:0%dense_mlp/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A Ж
 dense_mlp/BiasAdd/ReadVariableOpReadVariableOp)dense_mlp_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ъ
dense_mlp/BiasAddBiasAdddense_mlp/Tensordot:output:0(dense_mlp/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A Y
dense_mlp/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?К
dense_mlp/Gelu/mulMuldense_mlp/Gelu/mul/x:output:0dense_mlp/BiasAdd:output:0*
T0*+
_output_shapes
:         A Z
dense_mlp/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?У
dense_mlp/Gelu/truedivRealDivdense_mlp/BiasAdd:output:0dense_mlp/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A k
dense_mlp/Gelu/ErfErfdense_mlp/Gelu/truediv:z:0*
T0*+
_output_shapes
:         A Y
dense_mlp/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?И
dense_mlp/Gelu/addAddV2dense_mlp/Gelu/add/x:output:0dense_mlp/Gelu/Erf:y:0*
T0*+
_output_shapes
:         A Б
dense_mlp/Gelu/mul_1Muldense_mlp/Gelu/mul:z:0dense_mlp/Gelu/add:z:0*
T0*+
_output_shapes
:         A Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?К
dropout/dropout/MulMuldense_mlp/Gelu/mul_1:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:         A ]
dropout/dropout/ShapeShapedense_mlp/Gelu/mul_1:z:0*
T0*
_output_shapes
:а
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:         A *
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>┬
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A Г
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A Е
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:         A Ъ
(dense_embed_dim/Tensordot/ReadVariableOpReadVariableOp1dense_embed_dim_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0h
dense_embed_dim/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
dense_embed_dim/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       h
dense_embed_dim/Tensordot/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:i
'dense_embed_dim/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"dense_embed_dim/Tensordot/GatherV2GatherV2(dense_embed_dim/Tensordot/Shape:output:0'dense_embed_dim/Tensordot/free:output:00dense_embed_dim/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)dense_embed_dim/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$dense_embed_dim/Tensordot/GatherV2_1GatherV2(dense_embed_dim/Tensordot/Shape:output:0'dense_embed_dim/Tensordot/axes:output:02dense_embed_dim/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
dense_embed_dim/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
dense_embed_dim/Tensordot/ProdProd+dense_embed_dim/Tensordot/GatherV2:output:0(dense_embed_dim/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!dense_embed_dim/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 dense_embed_dim/Tensordot/Prod_1Prod-dense_embed_dim/Tensordot/GatherV2_1:output:0*dense_embed_dim/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%dense_embed_dim/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 dense_embed_dim/Tensordot/concatConcatV2'dense_embed_dim/Tensordot/free:output:0'dense_embed_dim/Tensordot/axes:output:0.dense_embed_dim/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
dense_embed_dim/Tensordot/stackPack'dense_embed_dim/Tensordot/Prod:output:0)dense_embed_dim/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:м
#dense_embed_dim/Tensordot/transpose	Transposedropout/dropout/Mul_1:z:0)dense_embed_dim/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A ║
!dense_embed_dim/Tensordot/ReshapeReshape'dense_embed_dim/Tensordot/transpose:y:0(dense_embed_dim/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 dense_embed_dim/Tensordot/MatMulMatMul*dense_embed_dim/Tensordot/Reshape:output:00dense_embed_dim/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @k
!dense_embed_dim/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@i
'dense_embed_dim/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"dense_embed_dim/Tensordot/concat_1ConcatV2+dense_embed_dim/Tensordot/GatherV2:output:0*dense_embed_dim/Tensordot/Const_2:output:00dense_embed_dim/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:│
dense_embed_dim/TensordotReshape*dense_embed_dim/Tensordot/MatMul:product:0+dense_embed_dim/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@Т
&dense_embed_dim/BiasAdd/ReadVariableOpReadVariableOp/dense_embed_dim_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
dense_embed_dim/BiasAddBiasAdd"dense_embed_dim/Tensordot:output:0.dense_embed_dim/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Ц
dropout_1/dropout/MulMul dense_embed_dim/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:         A@g
dropout_1/dropout/ShapeShape dense_embed_dim/BiasAdd:output:0*
T0*
_output_shapes
:д
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:         A@*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╚
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A@З
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A@Л
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:         A@n
IdentityIdentitydropout_1/dropout/Mul_1:z:0^NoOp*
T0*+
_output_shapes
:         A@т
NoOpNoOp'^dense_embed_dim/BiasAdd/ReadVariableOp)^dense_embed_dim/Tensordot/ReadVariableOp!^dense_mlp/BiasAdd/ReadVariableOp#^dense_mlp/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 2P
&dense_embed_dim/BiasAdd/ReadVariableOp&dense_embed_dim/BiasAdd/ReadVariableOp2T
(dense_embed_dim/Tensordot/ReadVariableOp(dense_embed_dim/Tensordot/ReadVariableOp2D
 dense_mlp/BiasAdd/ReadVariableOp dense_mlp/BiasAdd/ReadVariableOp2H
"dense_mlp/Tensordot/ReadVariableOp"dense_mlp/Tensordot/ReadVariableOp:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
л
═
*__inference_sequential_layer_call_fn_37946

inputs
unknown:@ 
	unknown_0: 
	unknown_1: @
	unknown_2:@
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_34559s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
┐О
У
L__inference_transformer_block_layer_call_and_return_conditional_losses_35637

inputsG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@S
Amulti_head_self_attention_query_tensordot_readvariableop_resource:@@M
?multi_head_self_attention_query_biasadd_readvariableop_resource:@Q
?multi_head_self_attention_key_tensordot_readvariableop_resource:@@K
=multi_head_self_attention_key_biasadd_readvariableop_resource:@S
Amulti_head_self_attention_value_tensordot_readvariableop_resource:@@M
?multi_head_self_attention_value_biasadd_readvariableop_resource:@Q
?multi_head_self_attention_out_tensordot_readvariableop_resource:@@K
=multi_head_self_attention_out_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@H
6sequential_dense_mlp_tensordot_readvariableop_resource:@ B
4sequential_dense_mlp_biasadd_readvariableop_resource: N
<sequential_dense_embed_dim_tensordot_readvariableop_resource: @H
:sequential_dense_embed_dim_biasadd_readvariableop_resource:@
identityИв,layer_normalization/batchnorm/ReadVariableOpв0layer_normalization/batchnorm/mul/ReadVariableOpв.layer_normalization_1/batchnorm/ReadVariableOpв2layer_normalization_1/batchnorm/mul/ReadVariableOpв4multi_head_self_attention/key/BiasAdd/ReadVariableOpв6multi_head_self_attention/key/Tensordot/ReadVariableOpв4multi_head_self_attention/out/BiasAdd/ReadVariableOpв6multi_head_self_attention/out/Tensordot/ReadVariableOpв6multi_head_self_attention/query/BiasAdd/ReadVariableOpв8multi_head_self_attention/query/Tensordot/ReadVariableOpв6multi_head_self_attention/value/BiasAdd/ReadVariableOpв8multi_head_self_attention/value/Tensordot/ReadVariableOpв1sequential/dense_embed_dim/BiasAdd/ReadVariableOpв3sequential/dense_embed_dim/Tensordot/ReadVariableOpв+sequential/dense_mlp/BiasAdd/ReadVariableOpв-sequential/dense_mlp/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┤
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(Щ
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         A│
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@А
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ч
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╜
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         AЙ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         Aж
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0┴
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@П
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@▓
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@Ю
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0╜
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@▓
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@v
multi_head_self_attention/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
-multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask║
8multi_head_self_attention/query/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_query_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0x
.multi_head_self_attention/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ж
/multi_head_self_attention/query/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:y
7multi_head_self_attention/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
2multi_head_self_attention/query/Tensordot/GatherV2GatherV28multi_head_self_attention/query/Tensordot/Shape:output:07multi_head_self_attention/query/Tensordot/free:output:0@multi_head_self_attention/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
4multi_head_self_attention/query/Tensordot/GatherV2_1GatherV28multi_head_self_attention/query/Tensordot/Shape:output:07multi_head_self_attention/query/Tensordot/axes:output:0Bmulti_head_self_attention/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/query/Tensordot/ProdProd;multi_head_self_attention/query/Tensordot/GatherV2:output:08multi_head_self_attention/query/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╘
0multi_head_self_attention/query/Tensordot/Prod_1Prod=multi_head_self_attention/query/Tensordot/GatherV2_1:output:0:multi_head_self_attention/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0multi_head_self_attention/query/Tensordot/concatConcatV27multi_head_self_attention/query/Tensordot/free:output:07multi_head_self_attention/query/Tensordot/axes:output:0>multi_head_self_attention/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┘
/multi_head_self_attention/query/Tensordot/stackPack7multi_head_self_attention/query/Tensordot/Prod:output:09multi_head_self_attention/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┌
3multi_head_self_attention/query/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:09multi_head_self_attention/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ъ
1multi_head_self_attention/query/Tensordot/ReshapeReshape7multi_head_self_attention/query/Tensordot/transpose:y:08multi_head_self_attention/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ъ
0multi_head_self_attention/query/Tensordot/MatMulMatMul:multi_head_self_attention/query/Tensordot/Reshape:output:0@multi_head_self_attention/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @{
1multi_head_self_attention/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@y
7multi_head_self_attention/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
2multi_head_self_attention/query/Tensordot/concat_1ConcatV2;multi_head_self_attention/query/Tensordot/GatherV2:output:0:multi_head_self_attention/query/Tensordot/Const_2:output:0@multi_head_self_attention/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:у
)multi_head_self_attention/query/TensordotReshape:multi_head_self_attention/query/Tensordot/MatMul:product:0;multi_head_self_attention/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@▓
6multi_head_self_attention/query/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_query_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▄
'multi_head_self_attention/query/BiasAddBiasAdd2multi_head_self_attention/query/Tensordot:output:0>multi_head_self_attention/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@╢
6multi_head_self_attention/key/Tensordot/ReadVariableOpReadVariableOp?multi_head_self_attention_key_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0v
,multi_head_self_attention/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,multi_head_self_attention/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Д
-multi_head_self_attention/key/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
5multi_head_self_attention/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : │
0multi_head_self_attention/key/Tensordot/GatherV2GatherV26multi_head_self_attention/key/Tensordot/Shape:output:05multi_head_self_attention/key/Tensordot/free:output:0>multi_head_self_attention/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7multi_head_self_attention/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╖
2multi_head_self_attention/key/Tensordot/GatherV2_1GatherV26multi_head_self_attention/key/Tensordot/Shape:output:05multi_head_self_attention/key/Tensordot/axes:output:0@multi_head_self_attention/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-multi_head_self_attention/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╚
,multi_head_self_attention/key/Tensordot/ProdProd9multi_head_self_attention/key/Tensordot/GatherV2:output:06multi_head_self_attention/key/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/multi_head_self_attention/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/key/Tensordot/Prod_1Prod;multi_head_self_attention/key/Tensordot/GatherV2_1:output:08multi_head_self_attention/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3multi_head_self_attention/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.multi_head_self_attention/key/Tensordot/concatConcatV25multi_head_self_attention/key/Tensordot/free:output:05multi_head_self_attention/key/Tensordot/axes:output:0<multi_head_self_attention/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╙
-multi_head_self_attention/key/Tensordot/stackPack5multi_head_self_attention/key/Tensordot/Prod:output:07multi_head_self_attention/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╓
1multi_head_self_attention/key/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:07multi_head_self_attention/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ф
/multi_head_self_attention/key/Tensordot/ReshapeReshape5multi_head_self_attention/key/Tensordot/transpose:y:06multi_head_self_attention/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ф
.multi_head_self_attention/key/Tensordot/MatMulMatMul8multi_head_self_attention/key/Tensordot/Reshape:output:0>multi_head_self_attention/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
/multi_head_self_attention/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@w
5multi_head_self_attention/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0multi_head_self_attention/key/Tensordot/concat_1ConcatV29multi_head_self_attention/key/Tensordot/GatherV2:output:08multi_head_self_attention/key/Tensordot/Const_2:output:0>multi_head_self_attention/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:▌
'multi_head_self_attention/key/TensordotReshape8multi_head_self_attention/key/Tensordot/MatMul:product:09multi_head_self_attention/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@о
4multi_head_self_attention/key/BiasAdd/ReadVariableOpReadVariableOp=multi_head_self_attention_key_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╓
%multi_head_self_attention/key/BiasAddBiasAdd0multi_head_self_attention/key/Tensordot:output:0<multi_head_self_attention/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@║
8multi_head_self_attention/value/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_value_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0x
.multi_head_self_attention/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ж
/multi_head_self_attention/value/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:y
7multi_head_self_attention/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
2multi_head_self_attention/value/Tensordot/GatherV2GatherV28multi_head_self_attention/value/Tensordot/Shape:output:07multi_head_self_attention/value/Tensordot/free:output:0@multi_head_self_attention/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
4multi_head_self_attention/value/Tensordot/GatherV2_1GatherV28multi_head_self_attention/value/Tensordot/Shape:output:07multi_head_self_attention/value/Tensordot/axes:output:0Bmulti_head_self_attention/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/value/Tensordot/ProdProd;multi_head_self_attention/value/Tensordot/GatherV2:output:08multi_head_self_attention/value/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╘
0multi_head_self_attention/value/Tensordot/Prod_1Prod=multi_head_self_attention/value/Tensordot/GatherV2_1:output:0:multi_head_self_attention/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0multi_head_self_attention/value/Tensordot/concatConcatV27multi_head_self_attention/value/Tensordot/free:output:07multi_head_self_attention/value/Tensordot/axes:output:0>multi_head_self_attention/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┘
/multi_head_self_attention/value/Tensordot/stackPack7multi_head_self_attention/value/Tensordot/Prod:output:09multi_head_self_attention/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┌
3multi_head_self_attention/value/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:09multi_head_self_attention/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ъ
1multi_head_self_attention/value/Tensordot/ReshapeReshape7multi_head_self_attention/value/Tensordot/transpose:y:08multi_head_self_attention/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ъ
0multi_head_self_attention/value/Tensordot/MatMulMatMul:multi_head_self_attention/value/Tensordot/Reshape:output:0@multi_head_self_attention/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @{
1multi_head_self_attention/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@y
7multi_head_self_attention/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
2multi_head_self_attention/value/Tensordot/concat_1ConcatV2;multi_head_self_attention/value/Tensordot/GatherV2:output:0:multi_head_self_attention/value/Tensordot/Const_2:output:0@multi_head_self_attention/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:у
)multi_head_self_attention/value/TensordotReshape:multi_head_self_attention/value/Tensordot/MatMul:product:0;multi_head_self_attention/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@▓
6multi_head_self_attention/value/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_value_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▄
'multi_head_self_attention/value/BiasAddBiasAdd2multi_head_self_attention/value/Tensordot:output:0>multi_head_self_attention/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@t
)multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         k
)multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
)multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╙
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/query/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"                  Б
(multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╥
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :│
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:╒
#multi_head_self_attention/Reshape_1Reshape.multi_head_self_attention/key/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╪
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :│
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:╫
#multi_head_self_attention/Reshape_2Reshape0multi_head_self_attention/value/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╪
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"                  ▐
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+                           *
adj_y(z
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:В
/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         {
1multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▌
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: k
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: ╟
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+                           Я
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+                           ╬
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╫
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@¤
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:╠
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :                  @╢
6multi_head_self_attention/out/Tensordot/ReadVariableOpReadVariableOp?multi_head_self_attention_out_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0v
,multi_head_self_attention/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,multi_head_self_attention/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Й
-multi_head_self_attention/out/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:w
5multi_head_self_attention/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : │
0multi_head_self_attention/out/Tensordot/GatherV2GatherV26multi_head_self_attention/out/Tensordot/Shape:output:05multi_head_self_attention/out/Tensordot/free:output:0>multi_head_self_attention/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7multi_head_self_attention/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╖
2multi_head_self_attention/out/Tensordot/GatherV2_1GatherV26multi_head_self_attention/out/Tensordot/Shape:output:05multi_head_self_attention/out/Tensordot/axes:output:0@multi_head_self_attention/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-multi_head_self_attention/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╚
,multi_head_self_attention/out/Tensordot/ProdProd9multi_head_self_attention/out/Tensordot/GatherV2:output:06multi_head_self_attention/out/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/multi_head_self_attention/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/out/Tensordot/Prod_1Prod;multi_head_self_attention/out/Tensordot/GatherV2_1:output:08multi_head_self_attention/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3multi_head_self_attention/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.multi_head_self_attention/out/Tensordot/concatConcatV25multi_head_self_attention/out/Tensordot/free:output:05multi_head_self_attention/out/Tensordot/axes:output:0<multi_head_self_attention/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╙
-multi_head_self_attention/out/Tensordot/stackPack5multi_head_self_attention/out/Tensordot/Prod:output:07multi_head_self_attention/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ф
1multi_head_self_attention/out/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:07multi_head_self_attention/out/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @ф
/multi_head_self_attention/out/Tensordot/ReshapeReshape5multi_head_self_attention/out/Tensordot/transpose:y:06multi_head_self_attention/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ф
.multi_head_self_attention/out/Tensordot/MatMulMatMul8multi_head_self_attention/out/Tensordot/Reshape:output:0>multi_head_self_attention/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
/multi_head_self_attention/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@w
5multi_head_self_attention/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0multi_head_self_attention/out/Tensordot/concat_1ConcatV29multi_head_self_attention/out/Tensordot/GatherV2:output:08multi_head_self_attention/out/Tensordot/Const_2:output:0>multi_head_self_attention/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ц
'multi_head_self_attention/out/TensordotReshape8multi_head_self_attention/out/Tensordot/MatMul:product:09multi_head_self_attention/out/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @о
4multi_head_self_attention/out/BiasAdd/ReadVariableOpReadVariableOp=multi_head_self_attention_out_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▀
%multi_head_self_attention/out/BiasAddBiasAdd0multi_head_self_attention/out/Tensordot:output:0<multi_head_self_attention/out/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?н
dropout_2/dropout/MulMul.multi_head_self_attention/out/BiasAdd:output:0 dropout_2/dropout/Const:output:0*
T0*4
_output_shapes"
 :                  @u
dropout_2/dropout/ShapeShape.multi_head_self_attention/out/BiasAdd:output:0*
T0*
_output_shapes
:н
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*4
_output_shapes"
 :                  @*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╤
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  @Р
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :                  @Ф
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*4
_output_shapes"
 :                  @g
addAddV2dropout_2/dropout/Mul_1:z:0inputs*
T0*+
_output_shapes
:         A@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╣
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(Э
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         A╕
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@В
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         AН
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         Aк
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╟
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@Ф
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@╕
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@в
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0├
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@╕
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@д
-sequential/dense_mlp/Tensordot/ReadVariableOpReadVariableOp6sequential_dense_mlp_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0m
#sequential/dense_mlp/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential/dense_mlp/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential/dense_mlp/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential/dense_mlp/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'sequential/dense_mlp/Tensordot/GatherV2GatherV2-sequential/dense_mlp/Tensordot/Shape:output:0,sequential/dense_mlp/Tensordot/free:output:05sequential/dense_mlp/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential/dense_mlp/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)sequential/dense_mlp/Tensordot/GatherV2_1GatherV2-sequential/dense_mlp/Tensordot/Shape:output:0,sequential/dense_mlp/Tensordot/axes:output:07sequential/dense_mlp/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential/dense_mlp/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
#sequential/dense_mlp/Tensordot/ProdProd0sequential/dense_mlp/Tensordot/GatherV2:output:0-sequential/dense_mlp/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential/dense_mlp/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%sequential/dense_mlp/Tensordot/Prod_1Prod2sequential/dense_mlp/Tensordot/GatherV2_1:output:0/sequential/dense_mlp/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential/dense_mlp/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%sequential/dense_mlp/Tensordot/concatConcatV2,sequential/dense_mlp/Tensordot/free:output:0,sequential/dense_mlp/Tensordot/axes:output:03sequential/dense_mlp/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$sequential/dense_mlp/Tensordot/stackPack,sequential/dense_mlp/Tensordot/Prod:output:0.sequential/dense_mlp/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╞
(sequential/dense_mlp/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0.sequential/dense_mlp/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@╔
&sequential/dense_mlp/Tensordot/ReshapeReshape,sequential/dense_mlp/Tensordot/transpose:y:0-sequential/dense_mlp/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%sequential/dense_mlp/Tensordot/MatMulMatMul/sequential/dense_mlp/Tensordot/Reshape:output:05sequential/dense_mlp/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          p
&sequential/dense_mlp/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: n
,sequential/dense_mlp/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'sequential/dense_mlp/Tensordot/concat_1ConcatV20sequential/dense_mlp/Tensordot/GatherV2:output:0/sequential/dense_mlp/Tensordot/Const_2:output:05sequential/dense_mlp/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┬
sequential/dense_mlp/TensordotReshape/sequential/dense_mlp/Tensordot/MatMul:product:00sequential/dense_mlp/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A Ь
+sequential/dense_mlp/BiasAdd/ReadVariableOpReadVariableOp4sequential_dense_mlp_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╗
sequential/dense_mlp/BiasAddBiasAdd'sequential/dense_mlp/Tensordot:output:03sequential/dense_mlp/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A d
sequential/dense_mlp/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?л
sequential/dense_mlp/Gelu/mulMul(sequential/dense_mlp/Gelu/mul/x:output:0%sequential/dense_mlp/BiasAdd:output:0*
T0*+
_output_shapes
:         A e
 sequential/dense_mlp/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?┤
!sequential/dense_mlp/Gelu/truedivRealDiv%sequential/dense_mlp/BiasAdd:output:0)sequential/dense_mlp/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A Б
sequential/dense_mlp/Gelu/ErfErf%sequential/dense_mlp/Gelu/truediv:z:0*
T0*+
_output_shapes
:         A d
sequential/dense_mlp/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?й
sequential/dense_mlp/Gelu/addAddV2(sequential/dense_mlp/Gelu/add/x:output:0!sequential/dense_mlp/Gelu/Erf:y:0*
T0*+
_output_shapes
:         A в
sequential/dense_mlp/Gelu/mul_1Mul!sequential/dense_mlp/Gelu/mul:z:0!sequential/dense_mlp/Gelu/add:z:0*
T0*+
_output_shapes
:         A e
 sequential/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?л
sequential/dropout/dropout/MulMul#sequential/dense_mlp/Gelu/mul_1:z:0)sequential/dropout/dropout/Const:output:0*
T0*+
_output_shapes
:         A s
 sequential/dropout/dropout/ShapeShape#sequential/dense_mlp/Gelu/mul_1:z:0*
T0*
_output_shapes
:╢
7sequential/dropout/dropout/random_uniform/RandomUniformRandomUniform)sequential/dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:         A *
dtype0n
)sequential/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>у
'sequential/dropout/dropout/GreaterEqualGreaterEqual@sequential/dropout/dropout/random_uniform/RandomUniform:output:02sequential/dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A Щ
sequential/dropout/dropout/CastCast+sequential/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A ж
 sequential/dropout/dropout/Mul_1Mul"sequential/dropout/dropout/Mul:z:0#sequential/dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:         A ░
3sequential/dense_embed_dim/Tensordot/ReadVariableOpReadVariableOp<sequential_dense_embed_dim_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0s
)sequential/dense_embed_dim/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)sequential/dense_embed_dim/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
*sequential/dense_embed_dim/Tensordot/ShapeShape$sequential/dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:t
2sequential/dense_embed_dim/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : з
-sequential/dense_embed_dim/Tensordot/GatherV2GatherV23sequential/dense_embed_dim/Tensordot/Shape:output:02sequential/dense_embed_dim/Tensordot/free:output:0;sequential/dense_embed_dim/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4sequential/dense_embed_dim/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : л
/sequential/dense_embed_dim/Tensordot/GatherV2_1GatherV23sequential/dense_embed_dim/Tensordot/Shape:output:02sequential/dense_embed_dim/Tensordot/axes:output:0=sequential/dense_embed_dim/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*sequential/dense_embed_dim/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ┐
)sequential/dense_embed_dim/Tensordot/ProdProd6sequential/dense_embed_dim/Tensordot/GatherV2:output:03sequential/dense_embed_dim/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,sequential/dense_embed_dim/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┼
+sequential/dense_embed_dim/Tensordot/Prod_1Prod8sequential/dense_embed_dim/Tensordot/GatherV2_1:output:05sequential/dense_embed_dim/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0sequential/dense_embed_dim/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : И
+sequential/dense_embed_dim/Tensordot/concatConcatV22sequential/dense_embed_dim/Tensordot/free:output:02sequential/dense_embed_dim/Tensordot/axes:output:09sequential/dense_embed_dim/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╩
*sequential/dense_embed_dim/Tensordot/stackPack2sequential/dense_embed_dim/Tensordot/Prod:output:04sequential/dense_embed_dim/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:═
.sequential/dense_embed_dim/Tensordot/transpose	Transpose$sequential/dropout/dropout/Mul_1:z:04sequential/dense_embed_dim/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A █
,sequential/dense_embed_dim/Tensordot/ReshapeReshape2sequential/dense_embed_dim/Tensordot/transpose:y:03sequential/dense_embed_dim/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  █
+sequential/dense_embed_dim/Tensordot/MatMulMatMul5sequential/dense_embed_dim/Tensordot/Reshape:output:0;sequential/dense_embed_dim/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @v
,sequential/dense_embed_dim/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2sequential/dense_embed_dim/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
-sequential/dense_embed_dim/Tensordot/concat_1ConcatV26sequential/dense_embed_dim/Tensordot/GatherV2:output:05sequential/dense_embed_dim/Tensordot/Const_2:output:0;sequential/dense_embed_dim/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╘
$sequential/dense_embed_dim/TensordotReshape5sequential/dense_embed_dim/Tensordot/MatMul:product:06sequential/dense_embed_dim/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@и
1sequential/dense_embed_dim/BiasAdd/ReadVariableOpReadVariableOp:sequential_dense_embed_dim_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0═
"sequential/dense_embed_dim/BiasAddBiasAdd-sequential/dense_embed_dim/Tensordot:output:09sequential/dense_embed_dim/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@g
"sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?╖
 sequential/dropout_1/dropout/MulMul+sequential/dense_embed_dim/BiasAdd:output:0+sequential/dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:         A@}
"sequential/dropout_1/dropout/ShapeShape+sequential/dense_embed_dim/BiasAdd:output:0*
T0*
_output_shapes
:║
9sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:         A@*
dtype0p
+sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>щ
)sequential/dropout_1/dropout/GreaterEqualGreaterEqualBsequential/dropout_1/dropout/random_uniform/RandomUniform:output:04sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A@Э
!sequential/dropout_1/dropout/CastCast-sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A@м
"sequential/dropout_1/dropout/Mul_1Mul$sequential/dropout_1/dropout/Mul:z:0%sequential/dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:         A@\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Ь
dropout_3/dropout/MulMul&sequential/dropout_1/dropout/Mul_1:z:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:         A@m
dropout_3/dropout/ShapeShape&sequential/dropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:д
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:         A@*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╚
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A@З
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A@Л
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:         A@j
add_1AddV2dropout_3/dropout/Mul_1:z:0add:z:0*
T0*+
_output_shapes
:         A@\
IdentityIdentity	add_1:z:0^NoOp*
T0*+
_output_shapes
:         A@Ю
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp5^multi_head_self_attention/key/BiasAdd/ReadVariableOp7^multi_head_self_attention/key/Tensordot/ReadVariableOp5^multi_head_self_attention/out/BiasAdd/ReadVariableOp7^multi_head_self_attention/out/Tensordot/ReadVariableOp7^multi_head_self_attention/query/BiasAdd/ReadVariableOp9^multi_head_self_attention/query/Tensordot/ReadVariableOp7^multi_head_self_attention/value/BiasAdd/ReadVariableOp9^multi_head_self_attention/value/Tensordot/ReadVariableOp2^sequential/dense_embed_dim/BiasAdd/ReadVariableOp4^sequential/dense_embed_dim/Tensordot/ReadVariableOp,^sequential/dense_mlp/BiasAdd/ReadVariableOp.^sequential/dense_mlp/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         A@: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2l
4multi_head_self_attention/key/BiasAdd/ReadVariableOp4multi_head_self_attention/key/BiasAdd/ReadVariableOp2p
6multi_head_self_attention/key/Tensordot/ReadVariableOp6multi_head_self_attention/key/Tensordot/ReadVariableOp2l
4multi_head_self_attention/out/BiasAdd/ReadVariableOp4multi_head_self_attention/out/BiasAdd/ReadVariableOp2p
6multi_head_self_attention/out/Tensordot/ReadVariableOp6multi_head_self_attention/out/Tensordot/ReadVariableOp2p
6multi_head_self_attention/query/BiasAdd/ReadVariableOp6multi_head_self_attention/query/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/query/Tensordot/ReadVariableOp8multi_head_self_attention/query/Tensordot/ReadVariableOp2p
6multi_head_self_attention/value/BiasAdd/ReadVariableOp6multi_head_self_attention/value/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/value/Tensordot/ReadVariableOp8multi_head_self_attention/value/Tensordot/ReadVariableOp2f
1sequential/dense_embed_dim/BiasAdd/ReadVariableOp1sequential/dense_embed_dim/BiasAdd/ReadVariableOp2j
3sequential/dense_embed_dim/Tensordot/ReadVariableOp3sequential/dense_embed_dim/Tensordot/ReadVariableOp2Z
+sequential/dense_mlp/BiasAdd/ReadVariableOp+sequential/dense_mlp/BiasAdd/ReadVariableOp2^
-sequential/dense_mlp/Tensordot/ReadVariableOp-sequential/dense_mlp/Tensordot/ReadVariableOp:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
л
═
*__inference_sequential_layer_call_fn_37933

inputs
unknown:@ 
	unknown_0: 
	unknown_1: @
	unknown_2:@
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_34451s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
Р

a
B__inference_dropout_layer_call_and_return_conditional_losses_38166

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         A C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         A *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         A ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         A "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A :S O
+
_output_shapes
:         A 
 
_user_specified_nameinputs
ц
▒
2__inference_vision_transformer_layer_call_fn_36233
x
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:A@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@ 

unknown_22: 

unknown_23: 


unknown_24:

identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_vision_transformer_layer_call_and_return_conditional_losses_35261o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:           

_user_specified_namex
В
b
)__inference_dropout_1_layer_call_fn_38215

inputs
identityИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_34482s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
╒
Ц
)__inference_dense_mlp_layer_call_fn_38101

inputs
unknown:@ 
	unknown_0: 
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_mlp_layer_call_and_return_conditional_losses_34394s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         A@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
╫
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_37889

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╠"
√
D__inference_dense_mlp_layer_call_and_return_conditional_losses_38139

inputs3
!tensordot_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         A@К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*+
_output_shapes
:         A P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?u
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A W
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:         A O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:         A c

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:         A a
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*+
_output_shapes
:         A z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         A@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
═
м
G__inference_sequential_1_layer_call_and_return_conditional_losses_34698

inputs)
layer_normalization_2_34645:@)
layer_normalization_2_34647:@
dense_1_34669:@ 
dense_1_34671: 
dense_2_34692: 

dense_2_34694:

identityИвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallд
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_2_34645layer_normalization_2_34647*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_34644Ь
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_1_34669dense_1_34671*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_34668▐
dropout_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_34679И
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_2_34692dense_2_34694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_34691w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
║
NoOpNoOp ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
░
E
)__inference_dropout_1_layer_call_fn_38210

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_34448d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         A@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A@:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
ч
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_38220

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         A@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         A@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A@:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
╘
╦
E__inference_sequential_layer_call_and_return_conditional_losses_34599
dense_mlp_input!
dense_mlp_34586:@ 
dense_mlp_34588: '
dense_embed_dim_34592: @#
dense_embed_dim_34594:@
identityИв'dense_embed_dim/StatefulPartitionedCallв!dense_mlp/StatefulPartitionedCallБ
!dense_mlp/StatefulPartitionedCallStatefulPartitionedCalldense_mlp_inputdense_mlp_34586dense_mlp_34588*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_mlp_layer_call_and_return_conditional_losses_34394р
dropout/PartitionedCallPartitionedCall*dense_mlp/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_34405к
'dense_embed_dim/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_embed_dim_34592dense_embed_dim_34594*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_34437ъ
dropout_1/PartitionedCallPartitionedCall0dense_embed_dim/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_34448u
IdentityIdentity"dropout_1/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@Ф
NoOpNoOp(^dense_embed_dim/StatefulPartitionedCall"^dense_mlp/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 2R
'dense_embed_dim/StatefulPartitionedCall'dense_embed_dim/StatefulPartitionedCall2F
!dense_mlp/StatefulPartitionedCall!dense_mlp/StatefulPartitionedCall:\ X
+
_output_shapes
:         A@
)
_user_specified_namedense_mlp_input
ё
Т
%__inference_dense_layer_call_fn_37051

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_34939|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╤
Б
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_38205

inputs3
!tensordot_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         A К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         A@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         A : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         A 
 
_user_specified_nameinputs
ею
Ч
M__inference_vision_transformer_layer_call_and_return_conditional_losses_37029
x9
'dense_tensordot_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@9
#broadcastto_readvariableop_resource:@1
add_readvariableop_resource:A@Y
Ktransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:@U
Gtransformer_block_layer_normalization_batchnorm_readvariableop_resource:@e
Stransformer_block_multi_head_self_attention_query_tensordot_readvariableop_resource:@@_
Qtransformer_block_multi_head_self_attention_query_biasadd_readvariableop_resource:@c
Qtransformer_block_multi_head_self_attention_key_tensordot_readvariableop_resource:@@]
Otransformer_block_multi_head_self_attention_key_biasadd_readvariableop_resource:@e
Stransformer_block_multi_head_self_attention_value_tensordot_readvariableop_resource:@@_
Qtransformer_block_multi_head_self_attention_value_biasadd_readvariableop_resource:@c
Qtransformer_block_multi_head_self_attention_out_tensordot_readvariableop_resource:@@]
Otransformer_block_multi_head_self_attention_out_biasadd_readvariableop_resource:@[
Mtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:@W
Itransformer_block_layer_normalization_1_batchnorm_readvariableop_resource:@Z
Htransformer_block_sequential_dense_mlp_tensordot_readvariableop_resource:@ T
Ftransformer_block_sequential_dense_mlp_biasadd_readvariableop_resource: `
Ntransformer_block_sequential_dense_embed_dim_tensordot_readvariableop_resource: @Z
Ltransformer_block_sequential_dense_embed_dim_biasadd_readvariableop_resource:@V
Hsequential_1_layer_normalization_2_batchnorm_mul_readvariableop_resource:@R
Dsequential_1_layer_normalization_2_batchnorm_readvariableop_resource:@E
3sequential_1_dense_1_matmul_readvariableop_resource:@ B
4sequential_1_dense_1_biasadd_readvariableop_resource: E
3sequential_1_dense_2_matmul_readvariableop_resource: 
B
4sequential_1_dense_2_biasadd_readvariableop_resource:

identityИвBroadcastTo/ReadVariableOpвadd/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/Tensordot/ReadVariableOpв+sequential_1/dense_1/BiasAdd/ReadVariableOpв*sequential_1/dense_1/MatMul/ReadVariableOpв+sequential_1/dense_2/BiasAdd/ReadVariableOpв*sequential_1/dense_2/MatMul/ReadVariableOpв;sequential_1/layer_normalization_2/batchnorm/ReadVariableOpв?sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOpв>transformer_block/layer_normalization/batchnorm/ReadVariableOpвBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpв@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpвDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpвFtransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpвHtransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpвFtransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpвHtransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpвHtransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpвJtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpвHtransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpвJtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpвCtransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpвEtransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpв=transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOpв?transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp6
ShapeShapex*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;W
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    l
rescaling/mulMulxrescaling/Cast/x:output:0*
T0*/
_output_shapes
:           А
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1/x:output:0*
T0*/
_output_shapes
:           H
Shape_1Shaperescaling/add:z:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╟
ExtractImagePatchesExtractImagePatchesrescaling/add:z:0*
T0*/
_output_shapes
:         *
ksizes
*
paddingVALID*
rates
*
strides
Z
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :С
Reshape/shapePackstrided_slice_1:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:И
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*4
_output_shapes"
 :                  Ж
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       U
dense/Tensordot/ShapeShapeReshape:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: А
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ж
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ┤
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Л
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ш
dense/Tensordot/transpose	TransposeReshape:output:0dense/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  Ь
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ь
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ю
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @В
BroadcastTo/ReadVariableOpReadVariableOp#broadcastto_readvariableop_resource*"
_output_shapes
:@*
dtype0U
BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :U
BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@Ы
BroadcastTo/shapePackstrided_slice:output:0BroadcastTo/shape/1:output:0BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:Р
BroadcastToBroadcastTo"BroadcastTo/ReadVariableOp:value:0BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         @M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ю
concatConcatV2BroadcastTo:output:0dense/BiasAdd:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  @r
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*"
_output_shapes
:A@*
dtype0o
addAddV2concat:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@О
Dtransformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┘
2transformer_block/layer_normalization/moments/meanMeanadd:z:0Mtransformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(╜
:transformer_block/layer_normalization/moments/StopGradientStopGradient;transformer_block/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         A╪
?transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferenceadd:z:0Ctransformer_block/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@Т
Htransformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Э
6transformer_block/layer_normalization/moments/varianceMeanCtransformer_block/layer_normalization/moments/SquaredDifference:z:0Qtransformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(z
5transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5є
3transformer_block/layer_normalization/batchnorm/addAddV2?transformer_block/layer_normalization/moments/variance:output:0>transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         Aн
5transformer_block/layer_normalization/batchnorm/RsqrtRsqrt7transformer_block/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         A╩
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpKtransformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ў
3transformer_block/layer_normalization/batchnorm/mulMul9transformer_block/layer_normalization/batchnorm/Rsqrt:y:0Jtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@┤
5transformer_block/layer_normalization/batchnorm/mul_1Muladd:z:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@ш
5transformer_block/layer_normalization/batchnorm/mul_2Mul;transformer_block/layer_normalization/moments/mean:output:07transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@┬
>transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpGtransformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0є
3transformer_block/layer_normalization/batchnorm/subSubFtransformer_block/layer_normalization/batchnorm/ReadVariableOp:value:09transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@ш
5transformer_block/layer_normalization/batchnorm/add_1AddV29transformer_block/layer_normalization/batchnorm/mul_1:z:07transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@Ъ
1transformer_block/multi_head_self_attention/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
Atransformer_block/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
Atransformer_block/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
9transformer_block/multi_head_self_attention/strided_sliceStridedSlice:transformer_block/multi_head_self_attention/Shape:output:0Htransformer_block/multi_head_self_attention/strided_slice/stack:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask▐
Jtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_query_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0К
@transformer_block/multi_head_self_attention/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:С
@transformer_block/multi_head_self_attention/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       к
Atransformer_block/multi_head_self_attention/query/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Л
Itransformer_block/multi_head_self_attention/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
Dtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2GatherV2Jtransformer_block/multi_head_self_attention/query/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/query/Tensordot/free:output:0Rtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Н
Ktransformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
Ftransformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1GatherV2Jtransformer_block/multi_head_self_attention/query/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/query/Tensordot/axes:output:0Ttransformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Л
Atransformer_block/multi_head_self_attention/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Д
@transformer_block/multi_head_self_attention/query/Tensordot/ProdProdMtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/query/Tensordot/Const:output:0*
T0*
_output_shapes
: Н
Ctransformer_block/multi_head_self_attention/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: К
Btransformer_block/multi_head_self_attention/query/Tensordot/Prod_1ProdOtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1:output:0Ltransformer_block/multi_head_self_attention/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Й
Gtransformer_block/multi_head_self_attention/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
Btransformer_block/multi_head_self_attention/query/Tensordot/concatConcatV2Itransformer_block/multi_head_self_attention/query/Tensordot/free:output:0Itransformer_block/multi_head_self_attention/query/Tensordot/axes:output:0Ptransformer_block/multi_head_self_attention/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:П
Atransformer_block/multi_head_self_attention/query/Tensordot/stackPackItransformer_block/multi_head_self_attention/query/Tensordot/Prod:output:0Ktransformer_block/multi_head_self_attention/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Р
Etransformer_block/multi_head_self_attention/query/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0Ktransformer_block/multi_head_self_attention/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@а
Ctransformer_block/multi_head_self_attention/query/Tensordot/ReshapeReshapeItransformer_block/multi_head_self_attention/query/Tensordot/transpose:y:0Jtransformer_block/multi_head_self_attention/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  а
Btransformer_block/multi_head_self_attention/query/Tensordot/MatMulMatMulLtransformer_block/multi_head_self_attention/query/Tensordot/Reshape:output:0Rtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Н
Ctransformer_block/multi_head_self_attention/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Л
Itransformer_block/multi_head_self_attention/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
Dtransformer_block/multi_head_self_attention/query/Tensordot/concat_1ConcatV2Mtransformer_block/multi_head_self_attention/query/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/query/Tensordot/Const_2:output:0Rtransformer_block/multi_head_self_attention/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Щ
;transformer_block/multi_head_self_attention/query/TensordotReshapeLtransformer_block/multi_head_self_attention/query/Tensordot/MatMul:product:0Mtransformer_block/multi_head_self_attention/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@╓
Htransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_query_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Т
9transformer_block/multi_head_self_attention/query/BiasAddBiasAddDtransformer_block/multi_head_self_attention/query/Tensordot:output:0Ptransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@┌
Htransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_key_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0И
>transformer_block/multi_head_self_attention/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:П
>transformer_block/multi_head_self_attention/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       и
?transformer_block/multi_head_self_attention/key/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Й
Gtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
Btransformer_block/multi_head_self_attention/key/Tensordot/GatherV2GatherV2Htransformer_block/multi_head_self_attention/key/Tensordot/Shape:output:0Gtransformer_block/multi_head_self_attention/key/Tensordot/free:output:0Ptransformer_block/multi_head_self_attention/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Л
Itransformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
Dtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1GatherV2Htransformer_block/multi_head_self_attention/key/Tensordot/Shape:output:0Gtransformer_block/multi_head_self_attention/key/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ■
>transformer_block/multi_head_self_attention/key/Tensordot/ProdProdKtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2:output:0Htransformer_block/multi_head_self_attention/key/Tensordot/Const:output:0*
T0*
_output_shapes
: Л
Atransformer_block/multi_head_self_attention/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Д
@transformer_block/multi_head_self_attention/key/Tensordot/Prod_1ProdMtransformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1:output:0Jtransformer_block/multi_head_self_attention/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: З
Etransformer_block/multi_head_self_attention/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
@transformer_block/multi_head_self_attention/key/Tensordot/concatConcatV2Gtransformer_block/multi_head_self_attention/key/Tensordot/free:output:0Gtransformer_block/multi_head_self_attention/key/Tensordot/axes:output:0Ntransformer_block/multi_head_self_attention/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/key/Tensordot/stackPackGtransformer_block/multi_head_self_attention/key/Tensordot/Prod:output:0Itransformer_block/multi_head_self_attention/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:М
Ctransformer_block/multi_head_self_attention/key/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0Itransformer_block/multi_head_self_attention/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@Ъ
Atransformer_block/multi_head_self_attention/key/Tensordot/ReshapeReshapeGtransformer_block/multi_head_self_attention/key/Tensordot/transpose:y:0Htransformer_block/multi_head_self_attention/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ъ
@transformer_block/multi_head_self_attention/key/Tensordot/MatMulMatMulJtransformer_block/multi_head_self_attention/key/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Л
Atransformer_block/multi_head_self_attention/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Й
Gtransformer_block/multi_head_self_attention/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Btransformer_block/multi_head_self_attention/key/Tensordot/concat_1ConcatV2Ktransformer_block/multi_head_self_attention/key/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/key/Tensordot/Const_2:output:0Ptransformer_block/multi_head_self_attention/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:У
9transformer_block/multi_head_self_attention/key/TensordotReshapeJtransformer_block/multi_head_self_attention/key/Tensordot/MatMul:product:0Ktransformer_block/multi_head_self_attention/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@╥
Ftransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpReadVariableOpOtransformer_block_multi_head_self_attention_key_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0М
7transformer_block/multi_head_self_attention/key/BiasAddBiasAddBtransformer_block/multi_head_self_attention/key/Tensordot:output:0Ntransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@▐
Jtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_value_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0К
@transformer_block/multi_head_self_attention/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:С
@transformer_block/multi_head_self_attention/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       к
Atransformer_block/multi_head_self_attention/value/Tensordot/ShapeShape9transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Л
Itransformer_block/multi_head_self_attention/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
Dtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2GatherV2Jtransformer_block/multi_head_self_attention/value/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/value/Tensordot/free:output:0Rtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Н
Ktransformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
Ftransformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1GatherV2Jtransformer_block/multi_head_self_attention/value/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/value/Tensordot/axes:output:0Ttransformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Л
Atransformer_block/multi_head_self_attention/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Д
@transformer_block/multi_head_self_attention/value/Tensordot/ProdProdMtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/value/Tensordot/Const:output:0*
T0*
_output_shapes
: Н
Ctransformer_block/multi_head_self_attention/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: К
Btransformer_block/multi_head_self_attention/value/Tensordot/Prod_1ProdOtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1:output:0Ltransformer_block/multi_head_self_attention/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Й
Gtransformer_block/multi_head_self_attention/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
Btransformer_block/multi_head_self_attention/value/Tensordot/concatConcatV2Itransformer_block/multi_head_self_attention/value/Tensordot/free:output:0Itransformer_block/multi_head_self_attention/value/Tensordot/axes:output:0Ptransformer_block/multi_head_self_attention/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:П
Atransformer_block/multi_head_self_attention/value/Tensordot/stackPackItransformer_block/multi_head_self_attention/value/Tensordot/Prod:output:0Ktransformer_block/multi_head_self_attention/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Р
Etransformer_block/multi_head_self_attention/value/Tensordot/transpose	Transpose9transformer_block/layer_normalization/batchnorm/add_1:z:0Ktransformer_block/multi_head_self_attention/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@а
Ctransformer_block/multi_head_self_attention/value/Tensordot/ReshapeReshapeItransformer_block/multi_head_self_attention/value/Tensordot/transpose:y:0Jtransformer_block/multi_head_self_attention/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  а
Btransformer_block/multi_head_self_attention/value/Tensordot/MatMulMatMulLtransformer_block/multi_head_self_attention/value/Tensordot/Reshape:output:0Rtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Н
Ctransformer_block/multi_head_self_attention/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Л
Itransformer_block/multi_head_self_attention/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
Dtransformer_block/multi_head_self_attention/value/Tensordot/concat_1ConcatV2Mtransformer_block/multi_head_self_attention/value/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/value/Tensordot/Const_2:output:0Rtransformer_block/multi_head_self_attention/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Щ
;transformer_block/multi_head_self_attention/value/TensordotReshapeLtransformer_block/multi_head_self_attention/value/Tensordot/MatMul:product:0Mtransformer_block/multi_head_self_attention/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@╓
Htransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_value_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Т
9transformer_block/multi_head_self_attention/value/BiasAddBiasAddDtransformer_block/multi_head_self_attention/value/Tensordot:output:0Ptransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@Ж
;transformer_block/multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         }
;transformer_block/multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :}
;transformer_block/multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Е
9transformer_block/multi_head_self_attention/Reshape/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/1:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/2:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:Й
3transformer_block/multi_head_self_attention/ReshapeReshapeBtransformer_block/multi_head_self_attention/query/BiasAdd:output:0Btransformer_block/multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"                  У
:transformer_block/multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             И
5transformer_block/multi_head_self_attention/transpose	Transpose<transformer_block/multi_head_self_attention/Reshape:output:0Ctransformer_block/multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"                  И
=transformer_block/multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         
=transformer_block/multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
=transformer_block/multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
;transformer_block/multi_head_self_attention/Reshape_1/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:Л
5transformer_block/multi_head_self_attention/Reshape_1Reshape@transformer_block/multi_head_self_attention/key/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"                  Х
<transformer_block/multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             О
7transformer_block/multi_head_self_attention/transpose_1	Transpose>transformer_block/multi_head_self_attention/Reshape_1:output:0Etransformer_block/multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"                  И
=transformer_block/multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         
=transformer_block/multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
=transformer_block/multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Н
;transformer_block/multi_head_self_attention/Reshape_2/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:Н
5transformer_block/multi_head_self_attention/Reshape_2ReshapeBtransformer_block/multi_head_self_attention/value/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"                  Х
<transformer_block/multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             О
7transformer_block/multi_head_self_attention/transpose_2	Transpose>transformer_block/multi_head_self_attention/Reshape_2:output:0Etransformer_block/multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"                  Ф
2transformer_block/multi_head_self_attention/MatMulBatchMatMulV29transformer_block/multi_head_self_attention/transpose:y:0;transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+                           *
adj_y(Ю
3transformer_block/multi_head_self_attention/Shape_1Shape;transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:Ф
Atransformer_block/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         Н
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Н
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╖
;transformer_block/multi_head_self_attention/strided_slice_1StridedSlice<transformer_block/multi_head_self_attention/Shape_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice_1/stack:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_1:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskо
0transformer_block/multi_head_self_attention/CastCastDtransformer_block/multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: П
0transformer_block/multi_head_self_attention/SqrtSqrt4transformer_block/multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: ¤
3transformer_block/multi_head_self_attention/truedivRealDiv;transformer_block/multi_head_self_attention/MatMul:output:04transformer_block/multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+                           ├
3transformer_block/multi_head_self_attention/SoftmaxSoftmax7transformer_block/multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+                           Д
4transformer_block/multi_head_self_attention/MatMul_1BatchMatMulV2=transformer_block/multi_head_self_attention/Softmax:softmax:0;transformer_block/multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"                  Х
<transformer_block/multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             Н
7transformer_block/multi_head_self_attention/transpose_3	Transpose=transformer_block/multi_head_self_attention/MatMul_1:output:0Etransformer_block/multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"                  И
=transformer_block/multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         
=transformer_block/multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@┼
;transformer_block/multi_head_self_attention/Reshape_3/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:В
5transformer_block/multi_head_self_attention/Reshape_3Reshape;transformer_block/multi_head_self_attention/transpose_3:y:0Dtransformer_block/multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :                  @┌
Htransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_out_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0И
>transformer_block/multi_head_self_attention/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:П
>transformer_block/multi_head_self_attention/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       н
?transformer_block/multi_head_self_attention/out/Tensordot/ShapeShape>transformer_block/multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:Й
Gtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
Btransformer_block/multi_head_self_attention/out/Tensordot/GatherV2GatherV2Htransformer_block/multi_head_self_attention/out/Tensordot/Shape:output:0Gtransformer_block/multi_head_self_attention/out/Tensordot/free:output:0Ptransformer_block/multi_head_self_attention/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Л
Itransformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
Dtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1GatherV2Htransformer_block/multi_head_self_attention/out/Tensordot/Shape:output:0Gtransformer_block/multi_head_self_attention/out/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ■
>transformer_block/multi_head_self_attention/out/Tensordot/ProdProdKtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2:output:0Htransformer_block/multi_head_self_attention/out/Tensordot/Const:output:0*
T0*
_output_shapes
: Л
Atransformer_block/multi_head_self_attention/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Д
@transformer_block/multi_head_self_attention/out/Tensordot/Prod_1ProdMtransformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1:output:0Jtransformer_block/multi_head_self_attention/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: З
Etransformer_block/multi_head_self_attention/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
@transformer_block/multi_head_self_attention/out/Tensordot/concatConcatV2Gtransformer_block/multi_head_self_attention/out/Tensordot/free:output:0Gtransformer_block/multi_head_self_attention/out/Tensordot/axes:output:0Ntransformer_block/multi_head_self_attention/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
?transformer_block/multi_head_self_attention/out/Tensordot/stackPackGtransformer_block/multi_head_self_attention/out/Tensordot/Prod:output:0Itransformer_block/multi_head_self_attention/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ъ
Ctransformer_block/multi_head_self_attention/out/Tensordot/transpose	Transpose>transformer_block/multi_head_self_attention/Reshape_3:output:0Itransformer_block/multi_head_self_attention/out/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @Ъ
Atransformer_block/multi_head_self_attention/out/Tensordot/ReshapeReshapeGtransformer_block/multi_head_self_attention/out/Tensordot/transpose:y:0Htransformer_block/multi_head_self_attention/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ъ
@transformer_block/multi_head_self_attention/out/Tensordot/MatMulMatMulJtransformer_block/multi_head_self_attention/out/Tensordot/Reshape:output:0Ptransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Л
Atransformer_block/multi_head_self_attention/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Й
Gtransformer_block/multi_head_self_attention/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
Btransformer_block/multi_head_self_attention/out/Tensordot/concat_1ConcatV2Ktransformer_block/multi_head_self_attention/out/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/out/Tensordot/Const_2:output:0Ptransformer_block/multi_head_self_attention/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ь
9transformer_block/multi_head_self_attention/out/TensordotReshapeJtransformer_block/multi_head_self_attention/out/Tensordot/MatMul:product:0Ktransformer_block/multi_head_self_attention/out/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @╥
Ftransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpReadVariableOpOtransformer_block_multi_head_self_attention_out_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Х
7transformer_block/multi_head_self_attention/out/BiasAddBiasAddBtransformer_block/multi_head_self_attention/out/Tensordot:output:0Ntransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @n
)transformer_block/dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?у
'transformer_block/dropout_2/dropout/MulMul@transformer_block/multi_head_self_attention/out/BiasAdd:output:02transformer_block/dropout_2/dropout/Const:output:0*
T0*4
_output_shapes"
 :                  @Щ
)transformer_block/dropout_2/dropout/ShapeShape@transformer_block/multi_head_self_attention/out/BiasAdd:output:0*
T0*
_output_shapes
:╤
@transformer_block/dropout_2/dropout/random_uniform/RandomUniformRandomUniform2transformer_block/dropout_2/dropout/Shape:output:0*
T0*4
_output_shapes"
 :                  @*
dtype0w
2transformer_block/dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>З
0transformer_block/dropout_2/dropout/GreaterEqualGreaterEqualItransformer_block/dropout_2/dropout/random_uniform/RandomUniform:output:0;transformer_block/dropout_2/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  @┤
(transformer_block/dropout_2/dropout/CastCast4transformer_block/dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :                  @╩
)transformer_block/dropout_2/dropout/Mul_1Mul+transformer_block/dropout_2/dropout/Mul:z:0,transformer_block/dropout_2/dropout/Cast:y:0*
T0*4
_output_shapes"
 :                  @М
transformer_block/addAddV2-transformer_block/dropout_2/dropout/Mul_1:z:0add:z:0*
T0*+
_output_shapes
:         A@Р
Ftransformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:я
4transformer_block/layer_normalization_1/moments/meanMeantransformer_block/add:z:0Otransformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(┴
<transformer_block/layer_normalization_1/moments/StopGradientStopGradient=transformer_block/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         Aю
Atransformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifferencetransformer_block/add:z:0Etransformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@Ф
Jtransformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:г
8transformer_block/layer_normalization_1/moments/varianceMeanEtransformer_block/layer_normalization_1/moments/SquaredDifference:z:0Stransformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(|
7transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5∙
5transformer_block/layer_normalization_1/batchnorm/addAddV2Atransformer_block/layer_normalization_1/moments/variance:output:0@transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         A▒
7transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrt9transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         A╬
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpMtransformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0¤
5transformer_block/layer_normalization_1/batchnorm/mulMul;transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0Ltransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@╩
7transformer_block/layer_normalization_1/batchnorm/mul_1Multransformer_block/add:z:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@ю
7transformer_block/layer_normalization_1/batchnorm/mul_2Mul=transformer_block/layer_normalization_1/moments/mean:output:09transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@╞
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpItransformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0∙
5transformer_block/layer_normalization_1/batchnorm/subSubHtransformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0;transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@ю
7transformer_block/layer_normalization_1/batchnorm/add_1AddV2;transformer_block/layer_normalization_1/batchnorm/mul_1:z:09transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@╚
?transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOpReadVariableOpHtransformer_block_sequential_dense_mlp_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0
5transformer_block/sequential/dense_mlp/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Ж
5transformer_block/sequential/dense_mlp/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       б
6transformer_block/sequential/dense_mlp/Tensordot/ShapeShape;transformer_block/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:А
>transformer_block/sequential/dense_mlp/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
9transformer_block/sequential/dense_mlp/Tensordot/GatherV2GatherV2?transformer_block/sequential/dense_mlp/Tensordot/Shape:output:0>transformer_block/sequential/dense_mlp/Tensordot/free:output:0Gtransformer_block/sequential/dense_mlp/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:В
@transformer_block/sequential/dense_mlp/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : █
;transformer_block/sequential/dense_mlp/Tensordot/GatherV2_1GatherV2?transformer_block/sequential/dense_mlp/Tensordot/Shape:output:0>transformer_block/sequential/dense_mlp/Tensordot/axes:output:0Itransformer_block/sequential/dense_mlp/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:А
6transformer_block/sequential/dense_mlp/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: у
5transformer_block/sequential/dense_mlp/Tensordot/ProdProdBtransformer_block/sequential/dense_mlp/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_mlp/Tensordot/Const:output:0*
T0*
_output_shapes
: В
8transformer_block/sequential/dense_mlp/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: щ
7transformer_block/sequential/dense_mlp/Tensordot/Prod_1ProdDtransformer_block/sequential/dense_mlp/Tensordot/GatherV2_1:output:0Atransformer_block/sequential/dense_mlp/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ~
<transformer_block/sequential/dense_mlp/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╕
7transformer_block/sequential/dense_mlp/Tensordot/concatConcatV2>transformer_block/sequential/dense_mlp/Tensordot/free:output:0>transformer_block/sequential/dense_mlp/Tensordot/axes:output:0Etransformer_block/sequential/dense_mlp/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ю
6transformer_block/sequential/dense_mlp/Tensordot/stackPack>transformer_block/sequential/dense_mlp/Tensordot/Prod:output:0@transformer_block/sequential/dense_mlp/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:№
:transformer_block/sequential/dense_mlp/Tensordot/transpose	Transpose;transformer_block/layer_normalization_1/batchnorm/add_1:z:0@transformer_block/sequential/dense_mlp/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ 
8transformer_block/sequential/dense_mlp/Tensordot/ReshapeReshape>transformer_block/sequential/dense_mlp/Tensordot/transpose:y:0?transformer_block/sequential/dense_mlp/Tensordot/stack:output:0*
T0*0
_output_shapes
:                   
7transformer_block/sequential/dense_mlp/Tensordot/MatMulMatMulAtransformer_block/sequential/dense_mlp/Tensordot/Reshape:output:0Gtransformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          В
8transformer_block/sequential/dense_mlp/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: А
>transformer_block/sequential/dense_mlp/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ├
9transformer_block/sequential/dense_mlp/Tensordot/concat_1ConcatV2Btransformer_block/sequential/dense_mlp/Tensordot/GatherV2:output:0Atransformer_block/sequential/dense_mlp/Tensordot/Const_2:output:0Gtransformer_block/sequential/dense_mlp/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:°
0transformer_block/sequential/dense_mlp/TensordotReshapeAtransformer_block/sequential/dense_mlp/Tensordot/MatMul:product:0Btransformer_block/sequential/dense_mlp/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A └
=transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_mlp_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ё
.transformer_block/sequential/dense_mlp/BiasAddBiasAdd9transformer_block/sequential/dense_mlp/Tensordot:output:0Etransformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A v
1transformer_block/sequential/dense_mlp/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?с
/transformer_block/sequential/dense_mlp/Gelu/mulMul:transformer_block/sequential/dense_mlp/Gelu/mul/x:output:07transformer_block/sequential/dense_mlp/BiasAdd:output:0*
T0*+
_output_shapes
:         A w
2transformer_block/sequential/dense_mlp/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?ъ
3transformer_block/sequential/dense_mlp/Gelu/truedivRealDiv7transformer_block/sequential/dense_mlp/BiasAdd:output:0;transformer_block/sequential/dense_mlp/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A е
/transformer_block/sequential/dense_mlp/Gelu/ErfErf7transformer_block/sequential/dense_mlp/Gelu/truediv:z:0*
T0*+
_output_shapes
:         A v
1transformer_block/sequential/dense_mlp/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?▀
/transformer_block/sequential/dense_mlp/Gelu/addAddV2:transformer_block/sequential/dense_mlp/Gelu/add/x:output:03transformer_block/sequential/dense_mlp/Gelu/Erf:y:0*
T0*+
_output_shapes
:         A ╪
1transformer_block/sequential/dense_mlp/Gelu/mul_1Mul3transformer_block/sequential/dense_mlp/Gelu/mul:z:03transformer_block/sequential/dense_mlp/Gelu/add:z:0*
T0*+
_output_shapes
:         A w
2transformer_block/sequential/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?с
0transformer_block/sequential/dropout/dropout/MulMul5transformer_block/sequential/dense_mlp/Gelu/mul_1:z:0;transformer_block/sequential/dropout/dropout/Const:output:0*
T0*+
_output_shapes
:         A Ч
2transformer_block/sequential/dropout/dropout/ShapeShape5transformer_block/sequential/dense_mlp/Gelu/mul_1:z:0*
T0*
_output_shapes
:┌
Itransformer_block/sequential/dropout/dropout/random_uniform/RandomUniformRandomUniform;transformer_block/sequential/dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:         A *
dtype0А
;transformer_block/sequential/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>Щ
9transformer_block/sequential/dropout/dropout/GreaterEqualGreaterEqualRtransformer_block/sequential/dropout/dropout/random_uniform/RandomUniform:output:0Dtransformer_block/sequential/dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A ╜
1transformer_block/sequential/dropout/dropout/CastCast=transformer_block/sequential/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A ▄
2transformer_block/sequential/dropout/dropout/Mul_1Mul4transformer_block/sequential/dropout/dropout/Mul:z:05transformer_block/sequential/dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:         A ╘
Etransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpReadVariableOpNtransformer_block_sequential_dense_embed_dim_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0Е
;transformer_block/sequential/dense_embed_dim/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:М
;transformer_block/sequential/dense_embed_dim/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       в
<transformer_block/sequential/dense_embed_dim/Tensordot/ShapeShape6transformer_block/sequential/dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:Ж
Dtransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : я
?transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2GatherV2Etransformer_block/sequential/dense_embed_dim/Tensordot/Shape:output:0Dtransformer_block/sequential/dense_embed_dim/Tensordot/free:output:0Mtransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:И
Ftransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : є
Atransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1GatherV2Etransformer_block/sequential/dense_embed_dim/Tensordot/Shape:output:0Dtransformer_block/sequential/dense_embed_dim/Tensordot/axes:output:0Otransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ж
<transformer_block/sequential/dense_embed_dim/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ї
;transformer_block/sequential/dense_embed_dim/Tensordot/ProdProdHtransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2:output:0Etransformer_block/sequential/dense_embed_dim/Tensordot/Const:output:0*
T0*
_output_shapes
: И
>transformer_block/sequential/dense_embed_dim/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: √
=transformer_block/sequential/dense_embed_dim/Tensordot/Prod_1ProdJtransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1:output:0Gtransformer_block/sequential/dense_embed_dim/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Д
Btransformer_block/sequential/dense_embed_dim/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╨
=transformer_block/sequential/dense_embed_dim/Tensordot/concatConcatV2Dtransformer_block/sequential/dense_embed_dim/Tensordot/free:output:0Dtransformer_block/sequential/dense_embed_dim/Tensordot/axes:output:0Ktransformer_block/sequential/dense_embed_dim/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:А
<transformer_block/sequential/dense_embed_dim/Tensordot/stackPackDtransformer_block/sequential/dense_embed_dim/Tensordot/Prod:output:0Ftransformer_block/sequential/dense_embed_dim/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Г
@transformer_block/sequential/dense_embed_dim/Tensordot/transpose	Transpose6transformer_block/sequential/dropout/dropout/Mul_1:z:0Ftransformer_block/sequential/dense_embed_dim/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A С
>transformer_block/sequential/dense_embed_dim/Tensordot/ReshapeReshapeDtransformer_block/sequential/dense_embed_dim/Tensordot/transpose:y:0Etransformer_block/sequential/dense_embed_dim/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  С
=transformer_block/sequential/dense_embed_dim/Tensordot/MatMulMatMulGtransformer_block/sequential/dense_embed_dim/Tensordot/Reshape:output:0Mtransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @И
>transformer_block/sequential/dense_embed_dim/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Ж
Dtransformer_block/sequential/dense_embed_dim/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : █
?transformer_block/sequential/dense_embed_dim/Tensordot/concat_1ConcatV2Htransformer_block/sequential/dense_embed_dim/Tensordot/GatherV2:output:0Gtransformer_block/sequential/dense_embed_dim/Tensordot/Const_2:output:0Mtransformer_block/sequential/dense_embed_dim/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:К
6transformer_block/sequential/dense_embed_dim/TensordotReshapeGtransformer_block/sequential/dense_embed_dim/Tensordot/MatMul:product:0Htransformer_block/sequential/dense_embed_dim/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@╠
Ctransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpReadVariableOpLtransformer_block_sequential_dense_embed_dim_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Г
4transformer_block/sequential/dense_embed_dim/BiasAddBiasAdd?transformer_block/sequential/dense_embed_dim/Tensordot:output:0Ktransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@y
4transformer_block/sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?э
2transformer_block/sequential/dropout_1/dropout/MulMul=transformer_block/sequential/dense_embed_dim/BiasAdd:output:0=transformer_block/sequential/dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:         A@б
4transformer_block/sequential/dropout_1/dropout/ShapeShape=transformer_block/sequential/dense_embed_dim/BiasAdd:output:0*
T0*
_output_shapes
:▐
Ktransformer_block/sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform=transformer_block/sequential/dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:         A@*
dtype0В
=transformer_block/sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>Я
;transformer_block/sequential/dropout_1/dropout/GreaterEqualGreaterEqualTtransformer_block/sequential/dropout_1/dropout/random_uniform/RandomUniform:output:0Ftransformer_block/sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A@┴
3transformer_block/sequential/dropout_1/dropout/CastCast?transformer_block/sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A@т
4transformer_block/sequential/dropout_1/dropout/Mul_1Mul6transformer_block/sequential/dropout_1/dropout/Mul:z:07transformer_block/sequential/dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:         A@n
)transformer_block/dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?╥
'transformer_block/dropout_3/dropout/MulMul8transformer_block/sequential/dropout_1/dropout/Mul_1:z:02transformer_block/dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:         A@С
)transformer_block/dropout_3/dropout/ShapeShape8transformer_block/sequential/dropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:╚
@transformer_block/dropout_3/dropout/random_uniform/RandomUniformRandomUniform2transformer_block/dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:         A@*
dtype0w
2transformer_block/dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>■
0transformer_block/dropout_3/dropout/GreaterEqualGreaterEqualItransformer_block/dropout_3/dropout/random_uniform/RandomUniform:output:0;transformer_block/dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A@л
(transformer_block/dropout_3/dropout/CastCast4transformer_block/dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A@┴
)transformer_block/dropout_3/dropout/Mul_1Mul+transformer_block/dropout_3/dropout/Mul:z:0,transformer_block/dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:         A@а
transformer_block/add_1AddV2-transformer_block/dropout_3/dropout/Mul_1:z:0transformer_block/add:z:0*
T0*+
_output_shapes
:         A@f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Щ
strided_slice_2StridedSlicetransformer_block/add_1:z:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_maskЛ
Asequential_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:р
/sequential_1/layer_normalization_2/moments/meanMeanstrided_slice_2:output:0Jsequential_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(│
7sequential_1/layer_normalization_2/moments/StopGradientStopGradient8sequential_1/layer_normalization_2/moments/mean:output:0*
T0*'
_output_shapes
:         ▀
<sequential_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifferencestrided_slice_2:output:0@sequential_1/layer_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @П
Esequential_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Р
3sequential_1/layer_normalization_2/moments/varianceMean@sequential_1/layer_normalization_2/moments/SquaredDifference:z:0Nsequential_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(w
2sequential_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5ц
0sequential_1/layer_normalization_2/batchnorm/addAddV2<sequential_1/layer_normalization_2/moments/variance:output:0;sequential_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         г
2sequential_1/layer_normalization_2/batchnorm/RsqrtRsqrt4sequential_1/layer_normalization_2/batchnorm/add:z:0*
T0*'
_output_shapes
:         ─
?sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpHsequential_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0ъ
0sequential_1/layer_normalization_2/batchnorm/mulMul6sequential_1/layer_normalization_2/batchnorm/Rsqrt:y:0Gsequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╗
2sequential_1/layer_normalization_2/batchnorm/mul_1Mulstrided_slice_2:output:04sequential_1/layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @█
2sequential_1/layer_normalization_2/batchnorm/mul_2Mul8sequential_1/layer_normalization_2/moments/mean:output:04sequential_1/layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @╝
;sequential_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpDsequential_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0ц
0sequential_1/layer_normalization_2/batchnorm/subSubCsequential_1/layer_normalization_2/batchnorm/ReadVariableOp:value:06sequential_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*'
_output_shapes
:         @█
2sequential_1/layer_normalization_2/batchnorm/add_1AddV26sequential_1/layer_normalization_2/batchnorm/mul_1:z:04sequential_1/layer_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @Ю
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0├
sequential_1/dense_1/MatMulMatMul6sequential_1/layer_normalization_2/batchnorm/add_1:z:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Ь
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╡
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
sequential_1/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
sequential_1/dense_1/Gelu/mulMul(sequential_1/dense_1/Gelu/mul/x:output:0%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          e
 sequential_1/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?░
!sequential_1/dense_1/Gelu/truedivRealDiv%sequential_1/dense_1/BiasAdd:output:0)sequential_1/dense_1/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:          }
sequential_1/dense_1/Gelu/ErfErf%sequential_1/dense_1/Gelu/truediv:z:0*
T0*'
_output_shapes
:          d
sequential_1/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?е
sequential_1/dense_1/Gelu/addAddV2(sequential_1/dense_1/Gelu/add/x:output:0!sequential_1/dense_1/Gelu/Erf:y:0*
T0*'
_output_shapes
:          Ю
sequential_1/dense_1/Gelu/mul_1Mul!sequential_1/dense_1/Gelu/mul:z:0!sequential_1/dense_1/Gelu/add:z:0*
T0*'
_output_shapes
:          i
$sequential_1/dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?п
"sequential_1/dropout_4/dropout/MulMul#sequential_1/dense_1/Gelu/mul_1:z:0-sequential_1/dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:          w
$sequential_1/dropout_4/dropout/ShapeShape#sequential_1/dense_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:║
;sequential_1/dropout_4/dropout/random_uniform/RandomUniformRandomUniform-sequential_1/dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0r
-sequential_1/dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ы
+sequential_1/dropout_4/dropout/GreaterEqualGreaterEqualDsequential_1/dropout_4/dropout/random_uniform/RandomUniform:output:06sequential_1/dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          Э
#sequential_1/dropout_4/dropout/CastCast/sequential_1/dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          о
$sequential_1/dropout_4/dropout/Mul_1Mul&sequential_1/dropout_4/dropout/Mul:z:0'sequential_1/dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:          Ю
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0╡
sequential_1/dense_2/MatMulMatMul(sequential_1/dropout_4/dropout/Mul_1:z:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
Ь
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0╡
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
t
IdentityIdentity%sequential_1/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
ц
NoOpNoOp^BroadcastTo/ReadVariableOp^add/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp<^sequential_1/layer_normalization_2/batchnorm/ReadVariableOp@^sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?^transformer_block/layer_normalization/batchnorm/ReadVariableOpC^transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpA^transformer_block/layer_normalization_1/batchnorm/ReadVariableOpE^transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpG^transformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpI^transformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpG^transformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpI^transformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpI^transformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpK^transformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpI^transformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpK^transformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpD^transformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpF^transformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOp>^transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp@^transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 28
BroadcastTo/ReadVariableOpBroadcastTo/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2z
;sequential_1/layer_normalization_2/batchnorm/ReadVariableOp;sequential_1/layer_normalization_2/batchnorm/ReadVariableOp2В
?sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp?sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2А
>transformer_block/layer_normalization/batchnorm/ReadVariableOp>transformer_block/layer_normalization/batchnorm/ReadVariableOp2И
Btransformer_block/layer_normalization/batchnorm/mul/ReadVariableOpBtransformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2Д
@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp@transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2М
Dtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpDtransformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2Р
Ftransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpFtransformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOp2Ф
Htransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpHtransformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOp2Р
Ftransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpFtransformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOp2Ф
Htransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpHtransformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOp2Ф
Htransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpHtransformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOp2Ш
Jtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpJtransformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOp2Ф
Htransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpHtransformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOp2Ш
Jtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpJtransformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOp2К
Ctransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpCtransformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOp2О
Etransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpEtransformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOp2~
=transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp=transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp2В
?transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp?transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp:R N
/
_output_shapes
:           

_user_specified_namex
°
╖
2__inference_vision_transformer_layer_call_fn_35945
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:A@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@ 

unknown_22: 

unknown_23: 


unknown_24:

identityИвStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_vision_transformer_layer_call_and_return_conditional_losses_35833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_1
╡
П
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_37847

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:И
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:         З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:з
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5}
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*'
_output_shapes
:         ]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*'
_output_shapes
:         ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Б
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @r
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*'
_output_shapes
:         @v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0}
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*'
_output_shapes
:         @r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ўK
№
E__inference_sequential_layer_call_and_return_conditional_losses_38012

inputs=
+dense_mlp_tensordot_readvariableop_resource:@ 7
)dense_mlp_biasadd_readvariableop_resource: C
1dense_embed_dim_tensordot_readvariableop_resource: @=
/dense_embed_dim_biasadd_readvariableop_resource:@
identityИв&dense_embed_dim/BiasAdd/ReadVariableOpв(dense_embed_dim/Tensordot/ReadVariableOpв dense_mlp/BiasAdd/ReadVariableOpв"dense_mlp/Tensordot/ReadVariableOpО
"dense_mlp/Tensordot/ReadVariableOpReadVariableOp+dense_mlp_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0b
dense_mlp/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:i
dense_mlp/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       O
dense_mlp/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:c
!dense_mlp/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : у
dense_mlp/Tensordot/GatherV2GatherV2"dense_mlp/Tensordot/Shape:output:0!dense_mlp/Tensordot/free:output:0*dense_mlp/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
#dense_mlp/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
dense_mlp/Tensordot/GatherV2_1GatherV2"dense_mlp/Tensordot/Shape:output:0!dense_mlp/Tensordot/axes:output:0,dense_mlp/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
dense_mlp/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
dense_mlp/Tensordot/ProdProd%dense_mlp/Tensordot/GatherV2:output:0"dense_mlp/Tensordot/Const:output:0*
T0*
_output_shapes
: e
dense_mlp/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Т
dense_mlp/Tensordot/Prod_1Prod'dense_mlp/Tensordot/GatherV2_1:output:0$dense_mlp/Tensordot/Const_1:output:0*
T0*
_output_shapes
: a
dense_mlp/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ─
dense_mlp/Tensordot/concatConcatV2!dense_mlp/Tensordot/free:output:0!dense_mlp/Tensordot/axes:output:0(dense_mlp/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ч
dense_mlp/Tensordot/stackPack!dense_mlp/Tensordot/Prod:output:0#dense_mlp/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Н
dense_mlp/Tensordot/transpose	Transposeinputs#dense_mlp/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@и
dense_mlp/Tensordot/ReshapeReshape!dense_mlp/Tensordot/transpose:y:0"dense_mlp/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  и
dense_mlp/Tensordot/MatMulMatMul$dense_mlp/Tensordot/Reshape:output:0*dense_mlp/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          e
dense_mlp/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: c
!dense_mlp/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
dense_mlp/Tensordot/concat_1ConcatV2%dense_mlp/Tensordot/GatherV2:output:0$dense_mlp/Tensordot/Const_2:output:0*dense_mlp/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:б
dense_mlp/TensordotReshape$dense_mlp/Tensordot/MatMul:product:0%dense_mlp/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A Ж
 dense_mlp/BiasAdd/ReadVariableOpReadVariableOp)dense_mlp_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ъ
dense_mlp/BiasAddBiasAdddense_mlp/Tensordot:output:0(dense_mlp/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A Y
dense_mlp/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?К
dense_mlp/Gelu/mulMuldense_mlp/Gelu/mul/x:output:0dense_mlp/BiasAdd:output:0*
T0*+
_output_shapes
:         A Z
dense_mlp/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?У
dense_mlp/Gelu/truedivRealDivdense_mlp/BiasAdd:output:0dense_mlp/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A k
dense_mlp/Gelu/ErfErfdense_mlp/Gelu/truediv:z:0*
T0*+
_output_shapes
:         A Y
dense_mlp/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?И
dense_mlp/Gelu/addAddV2dense_mlp/Gelu/add/x:output:0dense_mlp/Gelu/Erf:y:0*
T0*+
_output_shapes
:         A Б
dense_mlp/Gelu/mul_1Muldense_mlp/Gelu/mul:z:0dense_mlp/Gelu/add:z:0*
T0*+
_output_shapes
:         A l
dropout/IdentityIdentitydense_mlp/Gelu/mul_1:z:0*
T0*+
_output_shapes
:         A Ъ
(dense_embed_dim/Tensordot/ReadVariableOpReadVariableOp1dense_embed_dim_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0h
dense_embed_dim/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
dense_embed_dim/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       h
dense_embed_dim/Tensordot/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:i
'dense_embed_dim/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : √
"dense_embed_dim/Tensordot/GatherV2GatherV2(dense_embed_dim/Tensordot/Shape:output:0'dense_embed_dim/Tensordot/free:output:00dense_embed_dim/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)dense_embed_dim/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B :  
$dense_embed_dim/Tensordot/GatherV2_1GatherV2(dense_embed_dim/Tensordot/Shape:output:0'dense_embed_dim/Tensordot/axes:output:02dense_embed_dim/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
dense_embed_dim/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ю
dense_embed_dim/Tensordot/ProdProd+dense_embed_dim/Tensordot/GatherV2:output:0(dense_embed_dim/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!dense_embed_dim/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: д
 dense_embed_dim/Tensordot/Prod_1Prod-dense_embed_dim/Tensordot/GatherV2_1:output:0*dense_embed_dim/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%dense_embed_dim/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▄
 dense_embed_dim/Tensordot/concatConcatV2'dense_embed_dim/Tensordot/free:output:0'dense_embed_dim/Tensordot/axes:output:0.dense_embed_dim/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:й
dense_embed_dim/Tensordot/stackPack'dense_embed_dim/Tensordot/Prod:output:0)dense_embed_dim/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:м
#dense_embed_dim/Tensordot/transpose	Transposedropout/Identity:output:0)dense_embed_dim/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A ║
!dense_embed_dim/Tensordot/ReshapeReshape'dense_embed_dim/Tensordot/transpose:y:0(dense_embed_dim/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ║
 dense_embed_dim/Tensordot/MatMulMatMul*dense_embed_dim/Tensordot/Reshape:output:00dense_embed_dim/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @k
!dense_embed_dim/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@i
'dense_embed_dim/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ч
"dense_embed_dim/Tensordot/concat_1ConcatV2+dense_embed_dim/Tensordot/GatherV2:output:0*dense_embed_dim/Tensordot/Const_2:output:00dense_embed_dim/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:│
dense_embed_dim/TensordotReshape*dense_embed_dim/Tensordot/MatMul:product:0+dense_embed_dim/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@Т
&dense_embed_dim/BiasAdd/ReadVariableOpReadVariableOp/dense_embed_dim_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
dense_embed_dim/BiasAddBiasAdd"dense_embed_dim/Tensordot:output:0.dense_embed_dim/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@v
dropout_1/IdentityIdentity dense_embed_dim/BiasAdd:output:0*
T0*+
_output_shapes
:         A@n
IdentityIdentitydropout_1/Identity:output:0^NoOp*
T0*+
_output_shapes
:         A@т
NoOpNoOp'^dense_embed_dim/BiasAdd/ReadVariableOp)^dense_embed_dim/Tensordot/ReadVariableOp!^dense_mlp/BiasAdd/ReadVariableOp#^dense_mlp/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 2P
&dense_embed_dim/BiasAdd/ReadVariableOp&dense_embed_dim/BiasAdd/ReadVariableOp2T
(dense_embed_dim/Tensordot/ReadVariableOp(dense_embed_dim/Tensordot/ReadVariableOp2D
 dense_mlp/BiasAdd/ReadVariableOp dense_mlp/BiasAdd/ReadVariableOp2H
"dense_mlp/Tensordot/ReadVariableOp"dense_mlp/Tensordot/ReadVariableOp:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
х
`
B__inference_dropout_layer_call_and_return_conditional_losses_38154

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         A _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         A "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A :S O
+
_output_shapes
:         A 
 
_user_specified_nameinputs
╠"
√
D__inference_dense_mlp_layer_call_and_return_conditional_losses_34394

inputs3
!tensordot_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         A@К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?l
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*+
_output_shapes
:         A P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?u
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A W
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:         A O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:         A c

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:         A a
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*+
_output_shapes
:         A z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         A@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
╛
С
E__inference_sequential_layer_call_and_return_conditional_losses_34615
dense_mlp_input!
dense_mlp_34602:@ 
dense_mlp_34604: '
dense_embed_dim_34608: @#
dense_embed_dim_34610:@
identityИв'dense_embed_dim/StatefulPartitionedCallв!dense_mlp/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallБ
!dense_mlp/StatefulPartitionedCallStatefulPartitionedCalldense_mlp_inputdense_mlp_34602dense_mlp_34604*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_mlp_layer_call_and_return_conditional_losses_34394Ё
dropout/StatefulPartitionedCallStatefulPartitionedCall*dense_mlp/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_34515▓
'dense_embed_dim/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_embed_dim_34608dense_embed_dim_34610*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_34437Ь
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall0dense_embed_dim/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_34482}
IdentityIdentity*dropout_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@┌
NoOpNoOp(^dense_embed_dim/StatefulPartitionedCall"^dense_mlp/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 2R
'dense_embed_dim/StatefulPartitionedCall'dense_embed_dim/StatefulPartitionedCall2F
!dense_mlp/StatefulPartitionedCall!dense_mlp/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:\ X
+
_output_shapes
:         A@
)
_user_specified_namedense_mlp_input
Є
b
)__inference_dropout_4_layer_call_fn_37884

inputs
identityИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_34743o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ч;
┬
G__inference_sequential_1_layer_call_and_return_conditional_losses_37208

inputsI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 
5
'dense_2_biasadd_readvariableop_resource:

identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpв.layer_normalization_2/batchnorm/ReadVariableOpв2layer_normalization_2/batchnorm/mul/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┤
"layer_normalization_2/moments/meanMeaninputs=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(Щ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*'
_output_shapes
:         │
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceinputs3layer_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @В
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:щ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5┐
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         Й
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*'
_output_shapes
:         к
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0├
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @П
%layer_normalization_2/batchnorm/mul_1Mulinputs'layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @┤
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @в
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0┐
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*'
_output_shapes
:         @┤
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Ь
dense_1/MatMulMatMul)layer_normalization_2/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          W
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?А
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          X
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Й
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:          c
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*'
_output_shapes
:          W
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*'
_output_shapes
:          w
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*'
_output_shapes
:          \
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?И
dropout_4/dropout/MulMuldense_1/Gelu/mul_1:z:0 dropout_4/dropout/Const:output:0*
T0*'
_output_shapes
:          ]
dropout_4/dropout/ShapeShapedense_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:а
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>─
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          Г
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          З
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*'
_output_shapes
:          Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0О
dense_2/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
о
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ь
`
D__inference_rescaling_layer_call_and_return_conditional_losses_37042

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ]
mulMulinputsCast/x:output:0*
T0*/
_output_shapes
:           b
addAddV2mul:z:0Cast_1/x:output:0*
T0*/
_output_shapes
:           W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:           :W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╣
┬
E__inference_sequential_layer_call_and_return_conditional_losses_34451

inputs!
dense_mlp_34395:@ 
dense_mlp_34397: '
dense_embed_dim_34438: @#
dense_embed_dim_34440:@
identityИв'dense_embed_dim/StatefulPartitionedCallв!dense_mlp/StatefulPartitionedCall°
!dense_mlp/StatefulPartitionedCallStatefulPartitionedCallinputsdense_mlp_34395dense_mlp_34397*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_mlp_layer_call_and_return_conditional_losses_34394р
dropout/PartitionedCallPartitionedCall*dense_mlp/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_34405к
'dense_embed_dim/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_embed_dim_34438dense_embed_dim_34440*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_34437ъ
dropout_1/PartitionedCallPartitionedCall0dense_embed_dim/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_34448u
IdentityIdentity"dropout_1/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@Ф
NoOpNoOp(^dense_embed_dim/StatefulPartitionedCall"^dense_mlp/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 2R
'dense_embed_dim/StatefulPartitionedCall'dense_embed_dim/StatefulPartitionedCall2F
!dense_mlp/StatefulPartitionedCall!dense_mlp/StatefulPartitionedCall:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
▓
є
B__inference_dense_1_layer_call_and_return_conditional_losses_34668

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          O

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?h
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:          P
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?q
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:          S
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:          O

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?f
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:          _

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:          ]
IdentityIdentityGelu/mul_1:z:0^NoOp*
T0*'
_output_shapes
:          w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Т

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_38232

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         A@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         A@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         A@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         A@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A@:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
 
ў
@__inference_dense_layer_call_and_return_conditional_losses_37081

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╡
П
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_34644

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:И
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:         З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:з
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5}
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*'
_output_shapes
:         ]
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*'
_output_shapes
:         ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Б
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @r
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*'
_output_shapes
:         @v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0}
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*'
_output_shapes
:         @r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         @А
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ь
`
D__inference_rescaling_layer_call_and_return_conditional_losses_34897

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ]
mulMulinputsCast/x:output:0*
T0*/
_output_shapes
:           b
addAddV2mul:z:0Cast_1/x:output:0*
T0*/
_output_shapes
:           W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:           :W S
/
_output_shapes
:           
 
_user_specified_nameinputs
┤
х
G__inference_sequential_1_layer_call_and_return_conditional_losses_34877
layer_normalization_2_input)
layer_normalization_2_34860:@)
layer_normalization_2_34862:@
dense_1_34865:@ 
dense_1_34867: 
dense_2_34871: 

dense_2_34873:

identityИвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCall╣
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_2_inputlayer_normalization_2_34860layer_normalization_2_34862*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_34644Ь
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_1_34865dense_1_34867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_34668ю
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_34743Р
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_2_34871dense_2_34873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_34691w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
▐
NoOpNoOp ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall:d `
'
_output_shapes
:         @
5
_user_specified_namelayer_normalization_2_input
Аэ
У
L__inference_transformer_block_layer_call_and_return_conditional_losses_37535

inputsG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@S
Amulti_head_self_attention_query_tensordot_readvariableop_resource:@@M
?multi_head_self_attention_query_biasadd_readvariableop_resource:@Q
?multi_head_self_attention_key_tensordot_readvariableop_resource:@@K
=multi_head_self_attention_key_biasadd_readvariableop_resource:@S
Amulti_head_self_attention_value_tensordot_readvariableop_resource:@@M
?multi_head_self_attention_value_biasadd_readvariableop_resource:@Q
?multi_head_self_attention_out_tensordot_readvariableop_resource:@@K
=multi_head_self_attention_out_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@H
6sequential_dense_mlp_tensordot_readvariableop_resource:@ B
4sequential_dense_mlp_biasadd_readvariableop_resource: N
<sequential_dense_embed_dim_tensordot_readvariableop_resource: @H
:sequential_dense_embed_dim_biasadd_readvariableop_resource:@
identityИв,layer_normalization/batchnorm/ReadVariableOpв0layer_normalization/batchnorm/mul/ReadVariableOpв.layer_normalization_1/batchnorm/ReadVariableOpв2layer_normalization_1/batchnorm/mul/ReadVariableOpв4multi_head_self_attention/key/BiasAdd/ReadVariableOpв6multi_head_self_attention/key/Tensordot/ReadVariableOpв4multi_head_self_attention/out/BiasAdd/ReadVariableOpв6multi_head_self_attention/out/Tensordot/ReadVariableOpв6multi_head_self_attention/query/BiasAdd/ReadVariableOpв8multi_head_self_attention/query/Tensordot/ReadVariableOpв6multi_head_self_attention/value/BiasAdd/ReadVariableOpв8multi_head_self_attention/value/Tensordot/ReadVariableOpв1sequential/dense_embed_dim/BiasAdd/ReadVariableOpв3sequential/dense_embed_dim/Tensordot/ReadVariableOpв+sequential/dense_mlp/BiasAdd/ReadVariableOpв-sequential/dense_mlp/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┤
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(Щ
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         A│
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@А
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ч
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╜
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         AЙ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         Aж
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0┴
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@П
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@▓
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@Ю
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0╜
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@▓
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@v
multi_head_self_attention/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
-multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask║
8multi_head_self_attention/query/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_query_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0x
.multi_head_self_attention/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ж
/multi_head_self_attention/query/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:y
7multi_head_self_attention/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
2multi_head_self_attention/query/Tensordot/GatherV2GatherV28multi_head_self_attention/query/Tensordot/Shape:output:07multi_head_self_attention/query/Tensordot/free:output:0@multi_head_self_attention/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
4multi_head_self_attention/query/Tensordot/GatherV2_1GatherV28multi_head_self_attention/query/Tensordot/Shape:output:07multi_head_self_attention/query/Tensordot/axes:output:0Bmulti_head_self_attention/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/query/Tensordot/ProdProd;multi_head_self_attention/query/Tensordot/GatherV2:output:08multi_head_self_attention/query/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╘
0multi_head_self_attention/query/Tensordot/Prod_1Prod=multi_head_self_attention/query/Tensordot/GatherV2_1:output:0:multi_head_self_attention/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0multi_head_self_attention/query/Tensordot/concatConcatV27multi_head_self_attention/query/Tensordot/free:output:07multi_head_self_attention/query/Tensordot/axes:output:0>multi_head_self_attention/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┘
/multi_head_self_attention/query/Tensordot/stackPack7multi_head_self_attention/query/Tensordot/Prod:output:09multi_head_self_attention/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┌
3multi_head_self_attention/query/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:09multi_head_self_attention/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ъ
1multi_head_self_attention/query/Tensordot/ReshapeReshape7multi_head_self_attention/query/Tensordot/transpose:y:08multi_head_self_attention/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ъ
0multi_head_self_attention/query/Tensordot/MatMulMatMul:multi_head_self_attention/query/Tensordot/Reshape:output:0@multi_head_self_attention/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @{
1multi_head_self_attention/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@y
7multi_head_self_attention/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
2multi_head_self_attention/query/Tensordot/concat_1ConcatV2;multi_head_self_attention/query/Tensordot/GatherV2:output:0:multi_head_self_attention/query/Tensordot/Const_2:output:0@multi_head_self_attention/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:у
)multi_head_self_attention/query/TensordotReshape:multi_head_self_attention/query/Tensordot/MatMul:product:0;multi_head_self_attention/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@▓
6multi_head_self_attention/query/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_query_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▄
'multi_head_self_attention/query/BiasAddBiasAdd2multi_head_self_attention/query/Tensordot:output:0>multi_head_self_attention/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@╢
6multi_head_self_attention/key/Tensordot/ReadVariableOpReadVariableOp?multi_head_self_attention_key_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0v
,multi_head_self_attention/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,multi_head_self_attention/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Д
-multi_head_self_attention/key/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
5multi_head_self_attention/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : │
0multi_head_self_attention/key/Tensordot/GatherV2GatherV26multi_head_self_attention/key/Tensordot/Shape:output:05multi_head_self_attention/key/Tensordot/free:output:0>multi_head_self_attention/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7multi_head_self_attention/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╖
2multi_head_self_attention/key/Tensordot/GatherV2_1GatherV26multi_head_self_attention/key/Tensordot/Shape:output:05multi_head_self_attention/key/Tensordot/axes:output:0@multi_head_self_attention/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-multi_head_self_attention/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╚
,multi_head_self_attention/key/Tensordot/ProdProd9multi_head_self_attention/key/Tensordot/GatherV2:output:06multi_head_self_attention/key/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/multi_head_self_attention/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/key/Tensordot/Prod_1Prod;multi_head_self_attention/key/Tensordot/GatherV2_1:output:08multi_head_self_attention/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3multi_head_self_attention/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.multi_head_self_attention/key/Tensordot/concatConcatV25multi_head_self_attention/key/Tensordot/free:output:05multi_head_self_attention/key/Tensordot/axes:output:0<multi_head_self_attention/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╙
-multi_head_self_attention/key/Tensordot/stackPack5multi_head_self_attention/key/Tensordot/Prod:output:07multi_head_self_attention/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╓
1multi_head_self_attention/key/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:07multi_head_self_attention/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ф
/multi_head_self_attention/key/Tensordot/ReshapeReshape5multi_head_self_attention/key/Tensordot/transpose:y:06multi_head_self_attention/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ф
.multi_head_self_attention/key/Tensordot/MatMulMatMul8multi_head_self_attention/key/Tensordot/Reshape:output:0>multi_head_self_attention/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
/multi_head_self_attention/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@w
5multi_head_self_attention/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0multi_head_self_attention/key/Tensordot/concat_1ConcatV29multi_head_self_attention/key/Tensordot/GatherV2:output:08multi_head_self_attention/key/Tensordot/Const_2:output:0>multi_head_self_attention/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:▌
'multi_head_self_attention/key/TensordotReshape8multi_head_self_attention/key/Tensordot/MatMul:product:09multi_head_self_attention/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@о
4multi_head_self_attention/key/BiasAdd/ReadVariableOpReadVariableOp=multi_head_self_attention_key_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╓
%multi_head_self_attention/key/BiasAddBiasAdd0multi_head_self_attention/key/Tensordot:output:0<multi_head_self_attention/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@║
8multi_head_self_attention/value/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_value_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0x
.multi_head_self_attention/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ж
/multi_head_self_attention/value/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:y
7multi_head_self_attention/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
2multi_head_self_attention/value/Tensordot/GatherV2GatherV28multi_head_self_attention/value/Tensordot/Shape:output:07multi_head_self_attention/value/Tensordot/free:output:0@multi_head_self_attention/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
4multi_head_self_attention/value/Tensordot/GatherV2_1GatherV28multi_head_self_attention/value/Tensordot/Shape:output:07multi_head_self_attention/value/Tensordot/axes:output:0Bmulti_head_self_attention/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/value/Tensordot/ProdProd;multi_head_self_attention/value/Tensordot/GatherV2:output:08multi_head_self_attention/value/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╘
0multi_head_self_attention/value/Tensordot/Prod_1Prod=multi_head_self_attention/value/Tensordot/GatherV2_1:output:0:multi_head_self_attention/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0multi_head_self_attention/value/Tensordot/concatConcatV27multi_head_self_attention/value/Tensordot/free:output:07multi_head_self_attention/value/Tensordot/axes:output:0>multi_head_self_attention/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┘
/multi_head_self_attention/value/Tensordot/stackPack7multi_head_self_attention/value/Tensordot/Prod:output:09multi_head_self_attention/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┌
3multi_head_self_attention/value/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:09multi_head_self_attention/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ъ
1multi_head_self_attention/value/Tensordot/ReshapeReshape7multi_head_self_attention/value/Tensordot/transpose:y:08multi_head_self_attention/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ъ
0multi_head_self_attention/value/Tensordot/MatMulMatMul:multi_head_self_attention/value/Tensordot/Reshape:output:0@multi_head_self_attention/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @{
1multi_head_self_attention/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@y
7multi_head_self_attention/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
2multi_head_self_attention/value/Tensordot/concat_1ConcatV2;multi_head_self_attention/value/Tensordot/GatherV2:output:0:multi_head_self_attention/value/Tensordot/Const_2:output:0@multi_head_self_attention/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:у
)multi_head_self_attention/value/TensordotReshape:multi_head_self_attention/value/Tensordot/MatMul:product:0;multi_head_self_attention/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@▓
6multi_head_self_attention/value/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_value_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▄
'multi_head_self_attention/value/BiasAddBiasAdd2multi_head_self_attention/value/Tensordot:output:0>multi_head_self_attention/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@t
)multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         k
)multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
)multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╙
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/query/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"                  Б
(multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╥
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :│
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:╒
#multi_head_self_attention/Reshape_1Reshape.multi_head_self_attention/key/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╪
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :│
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:╫
#multi_head_self_attention/Reshape_2Reshape0multi_head_self_attention/value/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╪
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"                  ▐
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+                           *
adj_y(z
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:В
/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         {
1multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▌
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: k
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: ╟
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+                           Я
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+                           ╬
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╫
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@¤
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:╠
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :                  @╢
6multi_head_self_attention/out/Tensordot/ReadVariableOpReadVariableOp?multi_head_self_attention_out_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0v
,multi_head_self_attention/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,multi_head_self_attention/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Й
-multi_head_self_attention/out/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:w
5multi_head_self_attention/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : │
0multi_head_self_attention/out/Tensordot/GatherV2GatherV26multi_head_self_attention/out/Tensordot/Shape:output:05multi_head_self_attention/out/Tensordot/free:output:0>multi_head_self_attention/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7multi_head_self_attention/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╖
2multi_head_self_attention/out/Tensordot/GatherV2_1GatherV26multi_head_self_attention/out/Tensordot/Shape:output:05multi_head_self_attention/out/Tensordot/axes:output:0@multi_head_self_attention/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-multi_head_self_attention/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╚
,multi_head_self_attention/out/Tensordot/ProdProd9multi_head_self_attention/out/Tensordot/GatherV2:output:06multi_head_self_attention/out/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/multi_head_self_attention/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/out/Tensordot/Prod_1Prod;multi_head_self_attention/out/Tensordot/GatherV2_1:output:08multi_head_self_attention/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3multi_head_self_attention/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.multi_head_self_attention/out/Tensordot/concatConcatV25multi_head_self_attention/out/Tensordot/free:output:05multi_head_self_attention/out/Tensordot/axes:output:0<multi_head_self_attention/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╙
-multi_head_self_attention/out/Tensordot/stackPack5multi_head_self_attention/out/Tensordot/Prod:output:07multi_head_self_attention/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ф
1multi_head_self_attention/out/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:07multi_head_self_attention/out/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @ф
/multi_head_self_attention/out/Tensordot/ReshapeReshape5multi_head_self_attention/out/Tensordot/transpose:y:06multi_head_self_attention/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ф
.multi_head_self_attention/out/Tensordot/MatMulMatMul8multi_head_self_attention/out/Tensordot/Reshape:output:0>multi_head_self_attention/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
/multi_head_self_attention/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@w
5multi_head_self_attention/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0multi_head_self_attention/out/Tensordot/concat_1ConcatV29multi_head_self_attention/out/Tensordot/GatherV2:output:08multi_head_self_attention/out/Tensordot/Const_2:output:0>multi_head_self_attention/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ц
'multi_head_self_attention/out/TensordotReshape8multi_head_self_attention/out/Tensordot/MatMul:product:09multi_head_self_attention/out/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @о
4multi_head_self_attention/out/BiasAdd/ReadVariableOpReadVariableOp=multi_head_self_attention_out_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▀
%multi_head_self_attention/out/BiasAddBiasAdd0multi_head_self_attention/out/Tensordot:output:0<multi_head_self_attention/out/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Н
dropout_2/IdentityIdentity.multi_head_self_attention/out/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @g
addAddV2dropout_2/Identity:output:0inputs*
T0*+
_output_shapes
:         A@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╣
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(Э
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         A╕
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@В
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         AН
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         Aк
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╟
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@Ф
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@╕
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@в
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0├
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@╕
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@д
-sequential/dense_mlp/Tensordot/ReadVariableOpReadVariableOp6sequential_dense_mlp_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0m
#sequential/dense_mlp/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential/dense_mlp/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential/dense_mlp/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential/dense_mlp/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'sequential/dense_mlp/Tensordot/GatherV2GatherV2-sequential/dense_mlp/Tensordot/Shape:output:0,sequential/dense_mlp/Tensordot/free:output:05sequential/dense_mlp/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential/dense_mlp/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)sequential/dense_mlp/Tensordot/GatherV2_1GatherV2-sequential/dense_mlp/Tensordot/Shape:output:0,sequential/dense_mlp/Tensordot/axes:output:07sequential/dense_mlp/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential/dense_mlp/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
#sequential/dense_mlp/Tensordot/ProdProd0sequential/dense_mlp/Tensordot/GatherV2:output:0-sequential/dense_mlp/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential/dense_mlp/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%sequential/dense_mlp/Tensordot/Prod_1Prod2sequential/dense_mlp/Tensordot/GatherV2_1:output:0/sequential/dense_mlp/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential/dense_mlp/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%sequential/dense_mlp/Tensordot/concatConcatV2,sequential/dense_mlp/Tensordot/free:output:0,sequential/dense_mlp/Tensordot/axes:output:03sequential/dense_mlp/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$sequential/dense_mlp/Tensordot/stackPack,sequential/dense_mlp/Tensordot/Prod:output:0.sequential/dense_mlp/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╞
(sequential/dense_mlp/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0.sequential/dense_mlp/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@╔
&sequential/dense_mlp/Tensordot/ReshapeReshape,sequential/dense_mlp/Tensordot/transpose:y:0-sequential/dense_mlp/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%sequential/dense_mlp/Tensordot/MatMulMatMul/sequential/dense_mlp/Tensordot/Reshape:output:05sequential/dense_mlp/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          p
&sequential/dense_mlp/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: n
,sequential/dense_mlp/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'sequential/dense_mlp/Tensordot/concat_1ConcatV20sequential/dense_mlp/Tensordot/GatherV2:output:0/sequential/dense_mlp/Tensordot/Const_2:output:05sequential/dense_mlp/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┬
sequential/dense_mlp/TensordotReshape/sequential/dense_mlp/Tensordot/MatMul:product:00sequential/dense_mlp/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A Ь
+sequential/dense_mlp/BiasAdd/ReadVariableOpReadVariableOp4sequential_dense_mlp_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╗
sequential/dense_mlp/BiasAddBiasAdd'sequential/dense_mlp/Tensordot:output:03sequential/dense_mlp/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A d
sequential/dense_mlp/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?л
sequential/dense_mlp/Gelu/mulMul(sequential/dense_mlp/Gelu/mul/x:output:0%sequential/dense_mlp/BiasAdd:output:0*
T0*+
_output_shapes
:         A e
 sequential/dense_mlp/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?┤
!sequential/dense_mlp/Gelu/truedivRealDiv%sequential/dense_mlp/BiasAdd:output:0)sequential/dense_mlp/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A Б
sequential/dense_mlp/Gelu/ErfErf%sequential/dense_mlp/Gelu/truediv:z:0*
T0*+
_output_shapes
:         A d
sequential/dense_mlp/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?й
sequential/dense_mlp/Gelu/addAddV2(sequential/dense_mlp/Gelu/add/x:output:0!sequential/dense_mlp/Gelu/Erf:y:0*
T0*+
_output_shapes
:         A в
sequential/dense_mlp/Gelu/mul_1Mul!sequential/dense_mlp/Gelu/mul:z:0!sequential/dense_mlp/Gelu/add:z:0*
T0*+
_output_shapes
:         A В
sequential/dropout/IdentityIdentity#sequential/dense_mlp/Gelu/mul_1:z:0*
T0*+
_output_shapes
:         A ░
3sequential/dense_embed_dim/Tensordot/ReadVariableOpReadVariableOp<sequential_dense_embed_dim_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0s
)sequential/dense_embed_dim/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)sequential/dense_embed_dim/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
*sequential/dense_embed_dim/Tensordot/ShapeShape$sequential/dropout/Identity:output:0*
T0*
_output_shapes
:t
2sequential/dense_embed_dim/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : з
-sequential/dense_embed_dim/Tensordot/GatherV2GatherV23sequential/dense_embed_dim/Tensordot/Shape:output:02sequential/dense_embed_dim/Tensordot/free:output:0;sequential/dense_embed_dim/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4sequential/dense_embed_dim/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : л
/sequential/dense_embed_dim/Tensordot/GatherV2_1GatherV23sequential/dense_embed_dim/Tensordot/Shape:output:02sequential/dense_embed_dim/Tensordot/axes:output:0=sequential/dense_embed_dim/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*sequential/dense_embed_dim/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ┐
)sequential/dense_embed_dim/Tensordot/ProdProd6sequential/dense_embed_dim/Tensordot/GatherV2:output:03sequential/dense_embed_dim/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,sequential/dense_embed_dim/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┼
+sequential/dense_embed_dim/Tensordot/Prod_1Prod8sequential/dense_embed_dim/Tensordot/GatherV2_1:output:05sequential/dense_embed_dim/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0sequential/dense_embed_dim/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : И
+sequential/dense_embed_dim/Tensordot/concatConcatV22sequential/dense_embed_dim/Tensordot/free:output:02sequential/dense_embed_dim/Tensordot/axes:output:09sequential/dense_embed_dim/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╩
*sequential/dense_embed_dim/Tensordot/stackPack2sequential/dense_embed_dim/Tensordot/Prod:output:04sequential/dense_embed_dim/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:═
.sequential/dense_embed_dim/Tensordot/transpose	Transpose$sequential/dropout/Identity:output:04sequential/dense_embed_dim/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A █
,sequential/dense_embed_dim/Tensordot/ReshapeReshape2sequential/dense_embed_dim/Tensordot/transpose:y:03sequential/dense_embed_dim/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  █
+sequential/dense_embed_dim/Tensordot/MatMulMatMul5sequential/dense_embed_dim/Tensordot/Reshape:output:0;sequential/dense_embed_dim/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @v
,sequential/dense_embed_dim/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2sequential/dense_embed_dim/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
-sequential/dense_embed_dim/Tensordot/concat_1ConcatV26sequential/dense_embed_dim/Tensordot/GatherV2:output:05sequential/dense_embed_dim/Tensordot/Const_2:output:0;sequential/dense_embed_dim/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╘
$sequential/dense_embed_dim/TensordotReshape5sequential/dense_embed_dim/Tensordot/MatMul:product:06sequential/dense_embed_dim/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@и
1sequential/dense_embed_dim/BiasAdd/ReadVariableOpReadVariableOp:sequential_dense_embed_dim_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0═
"sequential/dense_embed_dim/BiasAddBiasAdd-sequential/dense_embed_dim/Tensordot:output:09sequential/dense_embed_dim/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@М
sequential/dropout_1/IdentityIdentity+sequential/dense_embed_dim/BiasAdd:output:0*
T0*+
_output_shapes
:         A@|
dropout_3/IdentityIdentity&sequential/dropout_1/Identity:output:0*
T0*+
_output_shapes
:         A@j
add_1AddV2dropout_3/Identity:output:0add:z:0*
T0*+
_output_shapes
:         A@\
IdentityIdentity	add_1:z:0^NoOp*
T0*+
_output_shapes
:         A@Ю
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp5^multi_head_self_attention/key/BiasAdd/ReadVariableOp7^multi_head_self_attention/key/Tensordot/ReadVariableOp5^multi_head_self_attention/out/BiasAdd/ReadVariableOp7^multi_head_self_attention/out/Tensordot/ReadVariableOp7^multi_head_self_attention/query/BiasAdd/ReadVariableOp9^multi_head_self_attention/query/Tensordot/ReadVariableOp7^multi_head_self_attention/value/BiasAdd/ReadVariableOp9^multi_head_self_attention/value/Tensordot/ReadVariableOp2^sequential/dense_embed_dim/BiasAdd/ReadVariableOp4^sequential/dense_embed_dim/Tensordot/ReadVariableOp,^sequential/dense_mlp/BiasAdd/ReadVariableOp.^sequential/dense_mlp/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         A@: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2l
4multi_head_self_attention/key/BiasAdd/ReadVariableOp4multi_head_self_attention/key/BiasAdd/ReadVariableOp2p
6multi_head_self_attention/key/Tensordot/ReadVariableOp6multi_head_self_attention/key/Tensordot/ReadVariableOp2l
4multi_head_self_attention/out/BiasAdd/ReadVariableOp4multi_head_self_attention/out/BiasAdd/ReadVariableOp2p
6multi_head_self_attention/out/Tensordot/ReadVariableOp6multi_head_self_attention/out/Tensordot/ReadVariableOp2p
6multi_head_self_attention/query/BiasAdd/ReadVariableOp6multi_head_self_attention/query/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/query/Tensordot/ReadVariableOp8multi_head_self_attention/query/Tensordot/ReadVariableOp2p
6multi_head_self_attention/value/BiasAdd/ReadVariableOp6multi_head_self_attention/value/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/value/Tensordot/ReadVariableOp8multi_head_self_attention/value/Tensordot/ReadVariableOp2f
1sequential/dense_embed_dim/BiasAdd/ReadVariableOp1sequential/dense_embed_dim/BiasAdd/ReadVariableOp2j
3sequential/dense_embed_dim/Tensordot/ReadVariableOp3sequential/dense_embed_dim/Tensordot/ReadVariableOp2Z
+sequential/dense_mlp/BiasAdd/ReadVariableOp+sequential/dense_mlp/BiasAdd/ReadVariableOp2^
-sequential/dense_mlp/Tensordot/ReadVariableOp-sequential/dense_mlp/Tensordot/ReadVariableOp:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
╞
╓
*__inference_sequential_layer_call_fn_34583
dense_mlp_input
unknown:@ 
	unknown_0: 
	unknown_1: @
	unknown_2:@
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCalldense_mlp_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_34559s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         A@
)
_user_specified_namedense_mlp_input
┼	
є
B__inference_dense_2_layer_call_and_return_conditional_losses_37920

inputs0
matmul_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
г
И
E__inference_sequential_layer_call_and_return_conditional_losses_34559

inputs!
dense_mlp_34546:@ 
dense_mlp_34548: '
dense_embed_dim_34552: @#
dense_embed_dim_34554:@
identityИв'dense_embed_dim/StatefulPartitionedCallв!dense_mlp/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCall°
!dense_mlp/StatefulPartitionedCallStatefulPartitionedCallinputsdense_mlp_34546dense_mlp_34548*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_mlp_layer_call_and_return_conditional_losses_34394Ё
dropout/StatefulPartitionedCallStatefulPartitionedCall*dense_mlp/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_34515▓
'dense_embed_dim/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_embed_dim_34552dense_embed_dim_34554*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_34437Ь
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall0dense_embed_dim/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_34482}
IdentityIdentity*dropout_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@┌
NoOpNoOp(^dense_embed_dim/StatefulPartitionedCall"^dense_mlp/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 2R
'dense_embed_dim/StatefulPartitionedCall'dense_embed_dim/StatefulPartitionedCall2F
!dense_mlp/StatefulPartitionedCall!dense_mlp/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
ц
▒
2__inference_vision_transformer_layer_call_fn_36290
x
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:A@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@ 

unknown_22: 

unknown_23: 


unknown_24:

identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_vision_transformer_layer_call_and_return_conditional_losses_35833o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:           

_user_specified_namex
У=
Ц

M__inference_vision_transformer_layer_call_and_return_conditional_losses_36031
input_1
dense_35964:@
dense_35966:@9
#broadcastto_readvariableop_resource:@1
add_readvariableop_resource:A@%
transformer_block_35980:@%
transformer_block_35982:@)
transformer_block_35984:@@%
transformer_block_35986:@)
transformer_block_35988:@@%
transformer_block_35990:@)
transformer_block_35992:@@%
transformer_block_35994:@)
transformer_block_35996:@@%
transformer_block_35998:@%
transformer_block_36000:@%
transformer_block_36002:@)
transformer_block_36004:@ %
transformer_block_36006: )
transformer_block_36008: @%
transformer_block_36010:@ 
sequential_1_36017:@ 
sequential_1_36019:@$
sequential_1_36021:@  
sequential_1_36023: $
sequential_1_36025: 
 
sequential_1_36027:

identityИвBroadcastTo/ReadVariableOpвadd/ReadVariableOpвdense/StatefulPartitionedCallв$sequential_1/StatefulPartitionedCallв)transformer_block/StatefulPartitionedCall<
ShapeShapeinput_1*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┼
rescaling/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_34897Y
Shape_1Shape"rescaling/PartitionedCall:output:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╪
ExtractImagePatchesExtractImagePatches"rescaling/PartitionedCall:output:0*
T0*/
_output_shapes
:         *
ksizes
*
paddingVALID*
rates
*
strides
Z
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :С
Reshape/shapePackstrided_slice_1:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:И
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*4
_output_shapes"
 :                  √
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_35964dense_35966*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_34939В
BroadcastTo/ReadVariableOpReadVariableOp#broadcastto_readvariableop_resource*"
_output_shapes
:@*
dtype0U
BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :U
BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@Ы
BroadcastTo/shapePackstrided_slice:output:0BroadcastTo/shape/1:output:0BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:Р
BroadcastToBroadcastTo"BroadcastTo/ReadVariableOp:value:0BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         @M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :о
concatConcatV2BroadcastTo:output:0&dense/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  @r
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*"
_output_shapes
:A@*
dtype0o
addAddV2concat:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@У
)transformer_block/StatefulPartitionedCallStatefulPartitionedCalladd:z:0transformer_block_35980transformer_block_35982transformer_block_35984transformer_block_35986transformer_block_35988transformer_block_35990transformer_block_35992transformer_block_35994transformer_block_35996transformer_block_35998transformer_block_36000transformer_block_36002transformer_block_36004transformer_block_36006transformer_block_36008transformer_block_36010*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_35209f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ░
strided_slice_2StridedSlice2transformer_block/StatefulPartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_maskъ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0sequential_1_36017sequential_1_36019sequential_1_36021sequential_1_36023sequential_1_36025sequential_1_36027*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_34698|
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ы
NoOpNoOp^BroadcastTo/ReadVariableOp^add/ReadVariableOp^dense/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 28
BroadcastTo/ReadVariableOpBroadcastTo/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_1
Аэ
У
L__inference_transformer_block_layer_call_and_return_conditional_losses_35209

inputsG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@S
Amulti_head_self_attention_query_tensordot_readvariableop_resource:@@M
?multi_head_self_attention_query_biasadd_readvariableop_resource:@Q
?multi_head_self_attention_key_tensordot_readvariableop_resource:@@K
=multi_head_self_attention_key_biasadd_readvariableop_resource:@S
Amulti_head_self_attention_value_tensordot_readvariableop_resource:@@M
?multi_head_self_attention_value_biasadd_readvariableop_resource:@Q
?multi_head_self_attention_out_tensordot_readvariableop_resource:@@K
=multi_head_self_attention_out_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@H
6sequential_dense_mlp_tensordot_readvariableop_resource:@ B
4sequential_dense_mlp_biasadd_readvariableop_resource: N
<sequential_dense_embed_dim_tensordot_readvariableop_resource: @H
:sequential_dense_embed_dim_biasadd_readvariableop_resource:@
identityИв,layer_normalization/batchnorm/ReadVariableOpв0layer_normalization/batchnorm/mul/ReadVariableOpв.layer_normalization_1/batchnorm/ReadVariableOpв2layer_normalization_1/batchnorm/mul/ReadVariableOpв4multi_head_self_attention/key/BiasAdd/ReadVariableOpв6multi_head_self_attention/key/Tensordot/ReadVariableOpв4multi_head_self_attention/out/BiasAdd/ReadVariableOpв6multi_head_self_attention/out/Tensordot/ReadVariableOpв6multi_head_self_attention/query/BiasAdd/ReadVariableOpв8multi_head_self_attention/query/Tensordot/ReadVariableOpв6multi_head_self_attention/value/BiasAdd/ReadVariableOpв8multi_head_self_attention/value/Tensordot/ReadVariableOpв1sequential/dense_embed_dim/BiasAdd/ReadVariableOpв3sequential/dense_embed_dim/Tensordot/ReadVariableOpв+sequential/dense_mlp/BiasAdd/ReadVariableOpв-sequential/dense_mlp/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┤
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(Щ
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         A│
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@А
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ч
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╜
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         AЙ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         Aж
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0┴
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@П
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@▓
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@Ю
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0╜
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@▓
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@v
multi_head_self_attention/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
-multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask║
8multi_head_self_attention/query/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_query_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0x
.multi_head_self_attention/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ж
/multi_head_self_attention/query/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:y
7multi_head_self_attention/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
2multi_head_self_attention/query/Tensordot/GatherV2GatherV28multi_head_self_attention/query/Tensordot/Shape:output:07multi_head_self_attention/query/Tensordot/free:output:0@multi_head_self_attention/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
4multi_head_self_attention/query/Tensordot/GatherV2_1GatherV28multi_head_self_attention/query/Tensordot/Shape:output:07multi_head_self_attention/query/Tensordot/axes:output:0Bmulti_head_self_attention/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/query/Tensordot/ProdProd;multi_head_self_attention/query/Tensordot/GatherV2:output:08multi_head_self_attention/query/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╘
0multi_head_self_attention/query/Tensordot/Prod_1Prod=multi_head_self_attention/query/Tensordot/GatherV2_1:output:0:multi_head_self_attention/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0multi_head_self_attention/query/Tensordot/concatConcatV27multi_head_self_attention/query/Tensordot/free:output:07multi_head_self_attention/query/Tensordot/axes:output:0>multi_head_self_attention/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┘
/multi_head_self_attention/query/Tensordot/stackPack7multi_head_self_attention/query/Tensordot/Prod:output:09multi_head_self_attention/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┌
3multi_head_self_attention/query/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:09multi_head_self_attention/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ъ
1multi_head_self_attention/query/Tensordot/ReshapeReshape7multi_head_self_attention/query/Tensordot/transpose:y:08multi_head_self_attention/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ъ
0multi_head_self_attention/query/Tensordot/MatMulMatMul:multi_head_self_attention/query/Tensordot/Reshape:output:0@multi_head_self_attention/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @{
1multi_head_self_attention/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@y
7multi_head_self_attention/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
2multi_head_self_attention/query/Tensordot/concat_1ConcatV2;multi_head_self_attention/query/Tensordot/GatherV2:output:0:multi_head_self_attention/query/Tensordot/Const_2:output:0@multi_head_self_attention/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:у
)multi_head_self_attention/query/TensordotReshape:multi_head_self_attention/query/Tensordot/MatMul:product:0;multi_head_self_attention/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@▓
6multi_head_self_attention/query/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_query_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▄
'multi_head_self_attention/query/BiasAddBiasAdd2multi_head_self_attention/query/Tensordot:output:0>multi_head_self_attention/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@╢
6multi_head_self_attention/key/Tensordot/ReadVariableOpReadVariableOp?multi_head_self_attention_key_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0v
,multi_head_self_attention/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,multi_head_self_attention/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Д
-multi_head_self_attention/key/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
5multi_head_self_attention/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : │
0multi_head_self_attention/key/Tensordot/GatherV2GatherV26multi_head_self_attention/key/Tensordot/Shape:output:05multi_head_self_attention/key/Tensordot/free:output:0>multi_head_self_attention/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7multi_head_self_attention/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╖
2multi_head_self_attention/key/Tensordot/GatherV2_1GatherV26multi_head_self_attention/key/Tensordot/Shape:output:05multi_head_self_attention/key/Tensordot/axes:output:0@multi_head_self_attention/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-multi_head_self_attention/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╚
,multi_head_self_attention/key/Tensordot/ProdProd9multi_head_self_attention/key/Tensordot/GatherV2:output:06multi_head_self_attention/key/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/multi_head_self_attention/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/key/Tensordot/Prod_1Prod;multi_head_self_attention/key/Tensordot/GatherV2_1:output:08multi_head_self_attention/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3multi_head_self_attention/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.multi_head_self_attention/key/Tensordot/concatConcatV25multi_head_self_attention/key/Tensordot/free:output:05multi_head_self_attention/key/Tensordot/axes:output:0<multi_head_self_attention/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╙
-multi_head_self_attention/key/Tensordot/stackPack5multi_head_self_attention/key/Tensordot/Prod:output:07multi_head_self_attention/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╓
1multi_head_self_attention/key/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:07multi_head_self_attention/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ф
/multi_head_self_attention/key/Tensordot/ReshapeReshape5multi_head_self_attention/key/Tensordot/transpose:y:06multi_head_self_attention/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ф
.multi_head_self_attention/key/Tensordot/MatMulMatMul8multi_head_self_attention/key/Tensordot/Reshape:output:0>multi_head_self_attention/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
/multi_head_self_attention/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@w
5multi_head_self_attention/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0multi_head_self_attention/key/Tensordot/concat_1ConcatV29multi_head_self_attention/key/Tensordot/GatherV2:output:08multi_head_self_attention/key/Tensordot/Const_2:output:0>multi_head_self_attention/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:▌
'multi_head_self_attention/key/TensordotReshape8multi_head_self_attention/key/Tensordot/MatMul:product:09multi_head_self_attention/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@о
4multi_head_self_attention/key/BiasAdd/ReadVariableOpReadVariableOp=multi_head_self_attention_key_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╓
%multi_head_self_attention/key/BiasAddBiasAdd0multi_head_self_attention/key/Tensordot:output:0<multi_head_self_attention/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@║
8multi_head_self_attention/value/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_value_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0x
.multi_head_self_attention/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ж
/multi_head_self_attention/value/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:y
7multi_head_self_attention/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
2multi_head_self_attention/value/Tensordot/GatherV2GatherV28multi_head_self_attention/value/Tensordot/Shape:output:07multi_head_self_attention/value/Tensordot/free:output:0@multi_head_self_attention/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
4multi_head_self_attention/value/Tensordot/GatherV2_1GatherV28multi_head_self_attention/value/Tensordot/Shape:output:07multi_head_self_attention/value/Tensordot/axes:output:0Bmulti_head_self_attention/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/value/Tensordot/ProdProd;multi_head_self_attention/value/Tensordot/GatherV2:output:08multi_head_self_attention/value/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╘
0multi_head_self_attention/value/Tensordot/Prod_1Prod=multi_head_self_attention/value/Tensordot/GatherV2_1:output:0:multi_head_self_attention/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0multi_head_self_attention/value/Tensordot/concatConcatV27multi_head_self_attention/value/Tensordot/free:output:07multi_head_self_attention/value/Tensordot/axes:output:0>multi_head_self_attention/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┘
/multi_head_self_attention/value/Tensordot/stackPack7multi_head_self_attention/value/Tensordot/Prod:output:09multi_head_self_attention/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┌
3multi_head_self_attention/value/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:09multi_head_self_attention/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ъ
1multi_head_self_attention/value/Tensordot/ReshapeReshape7multi_head_self_attention/value/Tensordot/transpose:y:08multi_head_self_attention/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ъ
0multi_head_self_attention/value/Tensordot/MatMulMatMul:multi_head_self_attention/value/Tensordot/Reshape:output:0@multi_head_self_attention/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @{
1multi_head_self_attention/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@y
7multi_head_self_attention/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
2multi_head_self_attention/value/Tensordot/concat_1ConcatV2;multi_head_self_attention/value/Tensordot/GatherV2:output:0:multi_head_self_attention/value/Tensordot/Const_2:output:0@multi_head_self_attention/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:у
)multi_head_self_attention/value/TensordotReshape:multi_head_self_attention/value/Tensordot/MatMul:product:0;multi_head_self_attention/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@▓
6multi_head_self_attention/value/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_value_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▄
'multi_head_self_attention/value/BiasAddBiasAdd2multi_head_self_attention/value/Tensordot:output:0>multi_head_self_attention/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@t
)multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         k
)multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
)multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╙
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/query/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"                  Б
(multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╥
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :│
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:╒
#multi_head_self_attention/Reshape_1Reshape.multi_head_self_attention/key/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╪
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :│
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:╫
#multi_head_self_attention/Reshape_2Reshape0multi_head_self_attention/value/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╪
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"                  ▐
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+                           *
adj_y(z
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:В
/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         {
1multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▌
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: k
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: ╟
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+                           Я
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+                           ╬
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╫
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@¤
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:╠
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :                  @╢
6multi_head_self_attention/out/Tensordot/ReadVariableOpReadVariableOp?multi_head_self_attention_out_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0v
,multi_head_self_attention/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,multi_head_self_attention/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Й
-multi_head_self_attention/out/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:w
5multi_head_self_attention/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : │
0multi_head_self_attention/out/Tensordot/GatherV2GatherV26multi_head_self_attention/out/Tensordot/Shape:output:05multi_head_self_attention/out/Tensordot/free:output:0>multi_head_self_attention/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7multi_head_self_attention/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╖
2multi_head_self_attention/out/Tensordot/GatherV2_1GatherV26multi_head_self_attention/out/Tensordot/Shape:output:05multi_head_self_attention/out/Tensordot/axes:output:0@multi_head_self_attention/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-multi_head_self_attention/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╚
,multi_head_self_attention/out/Tensordot/ProdProd9multi_head_self_attention/out/Tensordot/GatherV2:output:06multi_head_self_attention/out/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/multi_head_self_attention/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/out/Tensordot/Prod_1Prod;multi_head_self_attention/out/Tensordot/GatherV2_1:output:08multi_head_self_attention/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3multi_head_self_attention/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.multi_head_self_attention/out/Tensordot/concatConcatV25multi_head_self_attention/out/Tensordot/free:output:05multi_head_self_attention/out/Tensordot/axes:output:0<multi_head_self_attention/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╙
-multi_head_self_attention/out/Tensordot/stackPack5multi_head_self_attention/out/Tensordot/Prod:output:07multi_head_self_attention/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ф
1multi_head_self_attention/out/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:07multi_head_self_attention/out/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @ф
/multi_head_self_attention/out/Tensordot/ReshapeReshape5multi_head_self_attention/out/Tensordot/transpose:y:06multi_head_self_attention/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ф
.multi_head_self_attention/out/Tensordot/MatMulMatMul8multi_head_self_attention/out/Tensordot/Reshape:output:0>multi_head_self_attention/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
/multi_head_self_attention/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@w
5multi_head_self_attention/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0multi_head_self_attention/out/Tensordot/concat_1ConcatV29multi_head_self_attention/out/Tensordot/GatherV2:output:08multi_head_self_attention/out/Tensordot/Const_2:output:0>multi_head_self_attention/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ц
'multi_head_self_attention/out/TensordotReshape8multi_head_self_attention/out/Tensordot/MatMul:product:09multi_head_self_attention/out/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @о
4multi_head_self_attention/out/BiasAdd/ReadVariableOpReadVariableOp=multi_head_self_attention_out_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▀
%multi_head_self_attention/out/BiasAddBiasAdd0multi_head_self_attention/out/Tensordot:output:0<multi_head_self_attention/out/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Н
dropout_2/IdentityIdentity.multi_head_self_attention/out/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @g
addAddV2dropout_2/Identity:output:0inputs*
T0*+
_output_shapes
:         A@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╣
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(Э
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         A╕
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@В
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         AН
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         Aк
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╟
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@Ф
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@╕
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@в
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0├
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@╕
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@д
-sequential/dense_mlp/Tensordot/ReadVariableOpReadVariableOp6sequential_dense_mlp_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0m
#sequential/dense_mlp/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential/dense_mlp/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential/dense_mlp/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential/dense_mlp/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'sequential/dense_mlp/Tensordot/GatherV2GatherV2-sequential/dense_mlp/Tensordot/Shape:output:0,sequential/dense_mlp/Tensordot/free:output:05sequential/dense_mlp/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential/dense_mlp/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)sequential/dense_mlp/Tensordot/GatherV2_1GatherV2-sequential/dense_mlp/Tensordot/Shape:output:0,sequential/dense_mlp/Tensordot/axes:output:07sequential/dense_mlp/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential/dense_mlp/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
#sequential/dense_mlp/Tensordot/ProdProd0sequential/dense_mlp/Tensordot/GatherV2:output:0-sequential/dense_mlp/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential/dense_mlp/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%sequential/dense_mlp/Tensordot/Prod_1Prod2sequential/dense_mlp/Tensordot/GatherV2_1:output:0/sequential/dense_mlp/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential/dense_mlp/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%sequential/dense_mlp/Tensordot/concatConcatV2,sequential/dense_mlp/Tensordot/free:output:0,sequential/dense_mlp/Tensordot/axes:output:03sequential/dense_mlp/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$sequential/dense_mlp/Tensordot/stackPack,sequential/dense_mlp/Tensordot/Prod:output:0.sequential/dense_mlp/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╞
(sequential/dense_mlp/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0.sequential/dense_mlp/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@╔
&sequential/dense_mlp/Tensordot/ReshapeReshape,sequential/dense_mlp/Tensordot/transpose:y:0-sequential/dense_mlp/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%sequential/dense_mlp/Tensordot/MatMulMatMul/sequential/dense_mlp/Tensordot/Reshape:output:05sequential/dense_mlp/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          p
&sequential/dense_mlp/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: n
,sequential/dense_mlp/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'sequential/dense_mlp/Tensordot/concat_1ConcatV20sequential/dense_mlp/Tensordot/GatherV2:output:0/sequential/dense_mlp/Tensordot/Const_2:output:05sequential/dense_mlp/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┬
sequential/dense_mlp/TensordotReshape/sequential/dense_mlp/Tensordot/MatMul:product:00sequential/dense_mlp/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A Ь
+sequential/dense_mlp/BiasAdd/ReadVariableOpReadVariableOp4sequential_dense_mlp_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╗
sequential/dense_mlp/BiasAddBiasAdd'sequential/dense_mlp/Tensordot:output:03sequential/dense_mlp/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A d
sequential/dense_mlp/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?л
sequential/dense_mlp/Gelu/mulMul(sequential/dense_mlp/Gelu/mul/x:output:0%sequential/dense_mlp/BiasAdd:output:0*
T0*+
_output_shapes
:         A e
 sequential/dense_mlp/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?┤
!sequential/dense_mlp/Gelu/truedivRealDiv%sequential/dense_mlp/BiasAdd:output:0)sequential/dense_mlp/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A Б
sequential/dense_mlp/Gelu/ErfErf%sequential/dense_mlp/Gelu/truediv:z:0*
T0*+
_output_shapes
:         A d
sequential/dense_mlp/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?й
sequential/dense_mlp/Gelu/addAddV2(sequential/dense_mlp/Gelu/add/x:output:0!sequential/dense_mlp/Gelu/Erf:y:0*
T0*+
_output_shapes
:         A в
sequential/dense_mlp/Gelu/mul_1Mul!sequential/dense_mlp/Gelu/mul:z:0!sequential/dense_mlp/Gelu/add:z:0*
T0*+
_output_shapes
:         A В
sequential/dropout/IdentityIdentity#sequential/dense_mlp/Gelu/mul_1:z:0*
T0*+
_output_shapes
:         A ░
3sequential/dense_embed_dim/Tensordot/ReadVariableOpReadVariableOp<sequential_dense_embed_dim_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0s
)sequential/dense_embed_dim/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)sequential/dense_embed_dim/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
*sequential/dense_embed_dim/Tensordot/ShapeShape$sequential/dropout/Identity:output:0*
T0*
_output_shapes
:t
2sequential/dense_embed_dim/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : з
-sequential/dense_embed_dim/Tensordot/GatherV2GatherV23sequential/dense_embed_dim/Tensordot/Shape:output:02sequential/dense_embed_dim/Tensordot/free:output:0;sequential/dense_embed_dim/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4sequential/dense_embed_dim/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : л
/sequential/dense_embed_dim/Tensordot/GatherV2_1GatherV23sequential/dense_embed_dim/Tensordot/Shape:output:02sequential/dense_embed_dim/Tensordot/axes:output:0=sequential/dense_embed_dim/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*sequential/dense_embed_dim/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ┐
)sequential/dense_embed_dim/Tensordot/ProdProd6sequential/dense_embed_dim/Tensordot/GatherV2:output:03sequential/dense_embed_dim/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,sequential/dense_embed_dim/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┼
+sequential/dense_embed_dim/Tensordot/Prod_1Prod8sequential/dense_embed_dim/Tensordot/GatherV2_1:output:05sequential/dense_embed_dim/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0sequential/dense_embed_dim/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : И
+sequential/dense_embed_dim/Tensordot/concatConcatV22sequential/dense_embed_dim/Tensordot/free:output:02sequential/dense_embed_dim/Tensordot/axes:output:09sequential/dense_embed_dim/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╩
*sequential/dense_embed_dim/Tensordot/stackPack2sequential/dense_embed_dim/Tensordot/Prod:output:04sequential/dense_embed_dim/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:═
.sequential/dense_embed_dim/Tensordot/transpose	Transpose$sequential/dropout/Identity:output:04sequential/dense_embed_dim/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A █
,sequential/dense_embed_dim/Tensordot/ReshapeReshape2sequential/dense_embed_dim/Tensordot/transpose:y:03sequential/dense_embed_dim/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  █
+sequential/dense_embed_dim/Tensordot/MatMulMatMul5sequential/dense_embed_dim/Tensordot/Reshape:output:0;sequential/dense_embed_dim/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @v
,sequential/dense_embed_dim/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2sequential/dense_embed_dim/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
-sequential/dense_embed_dim/Tensordot/concat_1ConcatV26sequential/dense_embed_dim/Tensordot/GatherV2:output:05sequential/dense_embed_dim/Tensordot/Const_2:output:0;sequential/dense_embed_dim/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╘
$sequential/dense_embed_dim/TensordotReshape5sequential/dense_embed_dim/Tensordot/MatMul:product:06sequential/dense_embed_dim/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@и
1sequential/dense_embed_dim/BiasAdd/ReadVariableOpReadVariableOp:sequential_dense_embed_dim_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0═
"sequential/dense_embed_dim/BiasAddBiasAdd-sequential/dense_embed_dim/Tensordot:output:09sequential/dense_embed_dim/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@М
sequential/dropout_1/IdentityIdentity+sequential/dense_embed_dim/BiasAdd:output:0*
T0*+
_output_shapes
:         A@|
dropout_3/IdentityIdentity&sequential/dropout_1/Identity:output:0*
T0*+
_output_shapes
:         A@j
add_1AddV2dropout_3/Identity:output:0add:z:0*
T0*+
_output_shapes
:         A@\
IdentityIdentity	add_1:z:0^NoOp*
T0*+
_output_shapes
:         A@Ю
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp5^multi_head_self_attention/key/BiasAdd/ReadVariableOp7^multi_head_self_attention/key/Tensordot/ReadVariableOp5^multi_head_self_attention/out/BiasAdd/ReadVariableOp7^multi_head_self_attention/out/Tensordot/ReadVariableOp7^multi_head_self_attention/query/BiasAdd/ReadVariableOp9^multi_head_self_attention/query/Tensordot/ReadVariableOp7^multi_head_self_attention/value/BiasAdd/ReadVariableOp9^multi_head_self_attention/value/Tensordot/ReadVariableOp2^sequential/dense_embed_dim/BiasAdd/ReadVariableOp4^sequential/dense_embed_dim/Tensordot/ReadVariableOp,^sequential/dense_mlp/BiasAdd/ReadVariableOp.^sequential/dense_mlp/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         A@: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2l
4multi_head_self_attention/key/BiasAdd/ReadVariableOp4multi_head_self_attention/key/BiasAdd/ReadVariableOp2p
6multi_head_self_attention/key/Tensordot/ReadVariableOp6multi_head_self_attention/key/Tensordot/ReadVariableOp2l
4multi_head_self_attention/out/BiasAdd/ReadVariableOp4multi_head_self_attention/out/BiasAdd/ReadVariableOp2p
6multi_head_self_attention/out/Tensordot/ReadVariableOp6multi_head_self_attention/out/Tensordot/ReadVariableOp2p
6multi_head_self_attention/query/BiasAdd/ReadVariableOp6multi_head_self_attention/query/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/query/Tensordot/ReadVariableOp8multi_head_self_attention/query/Tensordot/ReadVariableOp2p
6multi_head_self_attention/value/BiasAdd/ReadVariableOp6multi_head_self_attention/value/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/value/Tensordot/ReadVariableOp8multi_head_self_attention/value/Tensordot/ReadVariableOp2f
1sequential/dense_embed_dim/BiasAdd/ReadVariableOp1sequential/dense_embed_dim/BiasAdd/ReadVariableOp2j
3sequential/dense_embed_dim/Tensordot/ReadVariableOp3sequential/dense_embed_dim/Tensordot/ReadVariableOp2Z
+sequential/dense_mlp/BiasAdd/ReadVariableOp+sequential/dense_mlp/BiasAdd/ReadVariableOp2^
-sequential/dense_mlp/Tensordot/ReadVariableOp-sequential/dense_mlp/Tensordot/ReadVariableOp:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
╤x
э
!__inference__traced_restore_38445
file_prefix.
assignvariableop_pos_emb:A@2
assignvariableop_1_class_emb:@1
assignvariableop_2_dense_kernel:@+
assignvariableop_3_dense_bias:@]
Kassignvariableop_4_transformer_block_multi_head_self_attention_query_kernel:@@W
Iassignvariableop_5_transformer_block_multi_head_self_attention_query_bias:@[
Iassignvariableop_6_transformer_block_multi_head_self_attention_key_kernel:@@U
Gassignvariableop_7_transformer_block_multi_head_self_attention_key_bias:@]
Kassignvariableop_8_transformer_block_multi_head_self_attention_value_kernel:@@W
Iassignvariableop_9_transformer_block_multi_head_self_attention_value_bias:@\
Jassignvariableop_10_transformer_block_multi_head_self_attention_out_kernel:@@V
Hassignvariableop_11_transformer_block_multi_head_self_attention_out_bias:@6
$assignvariableop_12_dense_mlp_kernel:@ 0
"assignvariableop_13_dense_mlp_bias: <
*assignvariableop_14_dense_embed_dim_kernel: @6
(assignvariableop_15_dense_embed_dim_bias:@M
?assignvariableop_16_transformer_block_layer_normalization_gamma:@L
>assignvariableop_17_transformer_block_layer_normalization_beta:@O
Aassignvariableop_18_transformer_block_layer_normalization_1_gamma:@N
@assignvariableop_19_transformer_block_layer_normalization_1_beta:@=
/assignvariableop_20_layer_normalization_2_gamma:@<
.assignvariableop_21_layer_normalization_2_beta:@4
"assignvariableop_22_dense_1_kernel:@ .
 assignvariableop_23_dense_1_bias: 4
"assignvariableop_24_dense_2_kernel: 
.
 assignvariableop_25_dense_2_bias:
%
assignvariableop_26_total_1: %
assignvariableop_27_count_1: #
assignvariableop_28_total: #
assignvariableop_29_count: 
identity_31ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9¤

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*г

valueЩ
BЦ
B"pos_emb/.ATTRIBUTES/VARIABLE_VALUEB$class_emb/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHо
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ║
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Р
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOpAssignVariableOpassignvariableop_pos_embIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_1AssignVariableOpassignvariableop_1_class_embIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_4AssignVariableOpKassignvariableop_4_transformer_block_multi_head_self_attention_query_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_5AssignVariableOpIassignvariableop_5_transformer_block_multi_head_self_attention_query_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_6AssignVariableOpIassignvariableop_6_transformer_block_multi_head_self_attention_key_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_7AssignVariableOpGassignvariableop_7_transformer_block_multi_head_self_attention_key_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_8AssignVariableOpKassignvariableop_8_transformer_block_multi_head_self_attention_value_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_9AssignVariableOpIassignvariableop_9_transformer_block_multi_head_self_attention_value_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_10AssignVariableOpJassignvariableop_10_transformer_block_multi_head_self_attention_out_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_11AssignVariableOpHassignvariableop_11_transformer_block_multi_head_self_attention_out_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dense_mlp_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_mlp_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_14AssignVariableOp*assignvariableop_14_dense_embed_dim_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_15AssignVariableOp(assignvariableop_15_dense_embed_dim_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:░
AssignVariableOp_16AssignVariableOp?assignvariableop_16_transformer_block_layer_normalization_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_17AssignVariableOp>assignvariableop_17_transformer_block_layer_normalization_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_18AssignVariableOpAassignvariableop_18_transformer_block_layer_normalization_1_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_19AssignVariableOp@assignvariableop_19_transformer_block_layer_normalization_1_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_20AssignVariableOp/assignvariableop_20_layer_normalization_2_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_21AssignVariableOp.assignvariableop_21_layer_normalization_2_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 у
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: ╨
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
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
╞
╓
*__inference_sequential_layer_call_fn_34462
dense_mlp_input
unknown:@ 
	unknown_0: 
	unknown_1: @
	unknown_2:@
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCalldense_mlp_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_34451s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         A@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         A@
)
_user_specified_namedense_mlp_input
°
╖
2__inference_vision_transformer_layer_call_fn_35316
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:A@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@ 

unknown_22: 

unknown_23: 


unknown_24:

identityИвStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_vision_transformer_layer_call_and_return_conditional_losses_35261o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_1
╪3
┬
G__inference_sequential_1_layer_call_and_return_conditional_losses_37158

inputsI
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@ 5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 
5
'dense_2_biasadd_readvariableop_resource:

identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpв.layer_normalization_2/batchnorm/ReadVariableOpв2layer_normalization_2/batchnorm/mul/ReadVariableOp~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┤
"layer_normalization_2/moments/meanMeaninputs=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(Щ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*'
_output_shapes
:         │
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceinputs3layer_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @В
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:щ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5┐
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         Й
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*'
_output_shapes
:         к
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0├
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @П
%layer_normalization_2/batchnorm/mul_1Mulinputs'layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @┤
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @в
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0┐
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*'
_output_shapes
:         @┤
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0Ь
dense_1/MatMulMatMul)layer_normalization_2/batchnorm/add_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          W
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?А
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          X
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?Й
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:          c
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*'
_output_shapes
:          W
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*'
_output_shapes
:          w
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*'
_output_shapes
:          h
dropout_4/IdentityIdentitydense_1/Gelu/mul_1:z:0*
T0*'
_output_shapes
:          Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0О
dense_2/MatMulMatMuldropout_4/Identity:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
о
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ї
╨
G__inference_sequential_1_layer_call_and_return_conditional_losses_34805

inputs)
layer_normalization_2_34788:@)
layer_normalization_2_34790:@
dense_1_34793:@ 
dense_1_34795: 
dense_2_34799: 

dense_2_34801:

identityИвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCallд
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCallinputslayer_normalization_2_34788layer_normalization_2_34790*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_34644Ь
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_1_34793dense_1_34795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_34668ю
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_34743Р
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_2_34799dense_2_34801*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_34691w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
▐
NoOpNoOp ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
│
Х
1__inference_transformer_block_layer_call_fn_37245

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: @

unknown_14:@
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_35209s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         A@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
а
E
)__inference_dropout_4_layer_call_fn_37879

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_34679`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╤
Б
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_34437

inputs3
!tensordot_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         A К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         A@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         A : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         A 
 
_user_specified_nameinputs
с
Ь
/__inference_dense_embed_dim_layer_call_fn_38175

inputs
unknown: @
	unknown_0:@
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_34437s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         A : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         A 
 
_user_specified_nameinputs
Є	
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_34743

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
√<
Р

M__inference_vision_transformer_layer_call_and_return_conditional_losses_35261
x
dense_34940:@
dense_34942:@9
#broadcastto_readvariableop_resource:@1
add_readvariableop_resource:A@%
transformer_block_35210:@%
transformer_block_35212:@)
transformer_block_35214:@@%
transformer_block_35216:@)
transformer_block_35218:@@%
transformer_block_35220:@)
transformer_block_35222:@@%
transformer_block_35224:@)
transformer_block_35226:@@%
transformer_block_35228:@%
transformer_block_35230:@%
transformer_block_35232:@)
transformer_block_35234:@ %
transformer_block_35236: )
transformer_block_35238: @%
transformer_block_35240:@ 
sequential_1_35247:@ 
sequential_1_35249:@$
sequential_1_35251:@  
sequential_1_35253: $
sequential_1_35255: 
 
sequential_1_35257:

identityИвBroadcastTo/ReadVariableOpвadd/ReadVariableOpвdense/StatefulPartitionedCallв$sequential_1/StatefulPartitionedCallв)transformer_block/StatefulPartitionedCall6
ShapeShapex*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┐
rescaling/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_34897Y
Shape_1Shape"rescaling/PartitionedCall:output:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╪
ExtractImagePatchesExtractImagePatches"rescaling/PartitionedCall:output:0*
T0*/
_output_shapes
:         *
ksizes
*
paddingVALID*
rates
*
strides
Z
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :С
Reshape/shapePackstrided_slice_1:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:И
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*4
_output_shapes"
 :                  √
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_34940dense_34942*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_34939В
BroadcastTo/ReadVariableOpReadVariableOp#broadcastto_readvariableop_resource*"
_output_shapes
:@*
dtype0U
BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :U
BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@Ы
BroadcastTo/shapePackstrided_slice:output:0BroadcastTo/shape/1:output:0BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:Р
BroadcastToBroadcastTo"BroadcastTo/ReadVariableOp:value:0BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         @M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :о
concatConcatV2BroadcastTo:output:0&dense/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  @r
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*"
_output_shapes
:A@*
dtype0o
addAddV2concat:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@У
)transformer_block/StatefulPartitionedCallStatefulPartitionedCalladd:z:0transformer_block_35210transformer_block_35212transformer_block_35214transformer_block_35216transformer_block_35218transformer_block_35220transformer_block_35222transformer_block_35224transformer_block_35226transformer_block_35228transformer_block_35230transformer_block_35232transformer_block_35234transformer_block_35236transformer_block_35238transformer_block_35240*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_35209f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ░
strided_slice_2StridedSlice2transformer_block/StatefulPartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_maskъ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0sequential_1_35247sequential_1_35249sequential_1_35251sequential_1_35253sequential_1_35255sequential_1_35257*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_34698|
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ы
NoOpNoOp^BroadcastTo/ReadVariableOp^add/ReadVariableOp^dense/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 28
BroadcastTo/ReadVariableOpBroadcastTo/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:R N
/
_output_shapes
:           

_user_specified_namex
 
ў
@__inference_dense_layer_call_and_return_conditional_losses_34939

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
┴
Ф
'__inference_dense_1_layer_call_fn_37856

inputs
unknown:@ 
	unknown_0: 
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_34668o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
М
┴
G__inference_sequential_1_layer_call_and_return_conditional_losses_34857
layer_normalization_2_input)
layer_normalization_2_34840:@)
layer_normalization_2_34842:@
dense_1_34845:@ 
dense_1_34847: 
dense_2_34851: 

dense_2_34853:

identityИвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallв-layer_normalization_2/StatefulPartitionedCall╣
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_2_inputlayer_normalization_2_34840layer_normalization_2_34842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_34644Ь
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_1_34845dense_1_34847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_34668▐
dropout_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_34679И
dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_2_34851dense_2_34853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_34691w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
║
NoOpNoOp ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall:d `
'
_output_shapes
:         @
5
_user_specified_namelayer_normalization_2_input
о	
Ц
,__inference_sequential_1_layer_call_fn_34713
layer_normalization_2_input
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 

	unknown_4:

identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_34698o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
'
_output_shapes
:         @
5
_user_specified_namelayer_normalization_2_input
ч
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_34448

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         A@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         A@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A@:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
╫
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_34679

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:          [

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
│
Х
1__inference_transformer_block_layer_call_fn_37282

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: @

unknown_14:@
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_35637s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         A@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
└
E
)__inference_rescaling_layer_call_fn_37034

inputs
identity║
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_34897h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:           "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:           :W S
/
_output_shapes
:           
 
_user_specified_nameinputs
√<
Р

M__inference_vision_transformer_layer_call_and_return_conditional_losses_35833
x
dense_35766:@
dense_35768:@9
#broadcastto_readvariableop_resource:@1
add_readvariableop_resource:A@%
transformer_block_35782:@%
transformer_block_35784:@)
transformer_block_35786:@@%
transformer_block_35788:@)
transformer_block_35790:@@%
transformer_block_35792:@)
transformer_block_35794:@@%
transformer_block_35796:@)
transformer_block_35798:@@%
transformer_block_35800:@%
transformer_block_35802:@%
transformer_block_35804:@)
transformer_block_35806:@ %
transformer_block_35808: )
transformer_block_35810: @%
transformer_block_35812:@ 
sequential_1_35819:@ 
sequential_1_35821:@$
sequential_1_35823:@  
sequential_1_35825: $
sequential_1_35827: 
 
sequential_1_35829:

identityИвBroadcastTo/ReadVariableOpвadd/ReadVariableOpвdense/StatefulPartitionedCallв$sequential_1/StatefulPartitionedCallв)transformer_block/StatefulPartitionedCall6
ShapeShapex*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┐
rescaling/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_34897Y
Shape_1Shape"rescaling/PartitionedCall:output:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╪
ExtractImagePatchesExtractImagePatches"rescaling/PartitionedCall:output:0*
T0*/
_output_shapes
:         *
ksizes
*
paddingVALID*
rates
*
strides
Z
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :С
Reshape/shapePackstrided_slice_1:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:И
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*4
_output_shapes"
 :                  √
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_35766dense_35768*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_34939В
BroadcastTo/ReadVariableOpReadVariableOp#broadcastto_readvariableop_resource*"
_output_shapes
:@*
dtype0U
BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :U
BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@Ы
BroadcastTo/shapePackstrided_slice:output:0BroadcastTo/shape/1:output:0BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:Р
BroadcastToBroadcastTo"BroadcastTo/ReadVariableOp:value:0BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         @M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :о
concatConcatV2BroadcastTo:output:0&dense/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  @r
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*"
_output_shapes
:A@*
dtype0o
addAddV2concat:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@У
)transformer_block/StatefulPartitionedCallStatefulPartitionedCalladd:z:0transformer_block_35782transformer_block_35784transformer_block_35786transformer_block_35788transformer_block_35790transformer_block_35792transformer_block_35794transformer_block_35796transformer_block_35798transformer_block_35800transformer_block_35802transformer_block_35804transformer_block_35806transformer_block_35808transformer_block_35810transformer_block_35812*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_35637f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ░
strided_slice_2StridedSlice2transformer_block/StatefulPartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_maskъ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0sequential_1_35819sequential_1_35821sequential_1_35823sequential_1_35825sequential_1_35827sequential_1_35829*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_34805|
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ы
NoOpNoOp^BroadcastTo/ReadVariableOp^add/ReadVariableOp^dense/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 28
BroadcastTo/ReadVariableOpBroadcastTo/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:R N
/
_output_shapes
:           

_user_specified_namex
┐О
У
L__inference_transformer_block_layer_call_and_return_conditional_losses_37816

inputsG
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@S
Amulti_head_self_attention_query_tensordot_readvariableop_resource:@@M
?multi_head_self_attention_query_biasadd_readvariableop_resource:@Q
?multi_head_self_attention_key_tensordot_readvariableop_resource:@@K
=multi_head_self_attention_key_biasadd_readvariableop_resource:@S
Amulti_head_self_attention_value_tensordot_readvariableop_resource:@@M
?multi_head_self_attention_value_biasadd_readvariableop_resource:@Q
?multi_head_self_attention_out_tensordot_readvariableop_resource:@@K
=multi_head_self_attention_out_biasadd_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@H
6sequential_dense_mlp_tensordot_readvariableop_resource:@ B
4sequential_dense_mlp_biasadd_readvariableop_resource: N
<sequential_dense_embed_dim_tensordot_readvariableop_resource: @H
:sequential_dense_embed_dim_biasadd_readvariableop_resource:@
identityИв,layer_normalization/batchnorm/ReadVariableOpв0layer_normalization/batchnorm/mul/ReadVariableOpв.layer_normalization_1/batchnorm/ReadVariableOpв2layer_normalization_1/batchnorm/mul/ReadVariableOpв4multi_head_self_attention/key/BiasAdd/ReadVariableOpв6multi_head_self_attention/key/Tensordot/ReadVariableOpв4multi_head_self_attention/out/BiasAdd/ReadVariableOpв6multi_head_self_attention/out/Tensordot/ReadVariableOpв6multi_head_self_attention/query/BiasAdd/ReadVariableOpв8multi_head_self_attention/query/Tensordot/ReadVariableOpв6multi_head_self_attention/value/BiasAdd/ReadVariableOpв8multi_head_self_attention/value/Tensordot/ReadVariableOpв1sequential/dense_embed_dim/BiasAdd/ReadVariableOpв3sequential/dense_embed_dim/Tensordot/ReadVariableOpв+sequential/dense_mlp/BiasAdd/ReadVariableOpв-sequential/dense_mlp/Tensordot/ReadVariableOp|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:┤
 layer_normalization/moments/meanMeaninputs;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(Щ
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         A│
-layer_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@А
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ч
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5╜
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         AЙ
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         Aж
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0┴
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@П
#layer_normalization/batchnorm/mul_1Mulinputs%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@▓
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@Ю
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0╜
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@▓
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@v
multi_head_self_attention/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
-multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╙
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask║
8multi_head_self_attention/query/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_query_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0x
.multi_head_self_attention/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ж
/multi_head_self_attention/query/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:y
7multi_head_self_attention/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
2multi_head_self_attention/query/Tensordot/GatherV2GatherV28multi_head_self_attention/query/Tensordot/Shape:output:07multi_head_self_attention/query/Tensordot/free:output:0@multi_head_self_attention/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
4multi_head_self_attention/query/Tensordot/GatherV2_1GatherV28multi_head_self_attention/query/Tensordot/Shape:output:07multi_head_self_attention/query/Tensordot/axes:output:0Bmulti_head_self_attention/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/query/Tensordot/ProdProd;multi_head_self_attention/query/Tensordot/GatherV2:output:08multi_head_self_attention/query/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╘
0multi_head_self_attention/query/Tensordot/Prod_1Prod=multi_head_self_attention/query/Tensordot/GatherV2_1:output:0:multi_head_self_attention/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0multi_head_self_attention/query/Tensordot/concatConcatV27multi_head_self_attention/query/Tensordot/free:output:07multi_head_self_attention/query/Tensordot/axes:output:0>multi_head_self_attention/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┘
/multi_head_self_attention/query/Tensordot/stackPack7multi_head_self_attention/query/Tensordot/Prod:output:09multi_head_self_attention/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┌
3multi_head_self_attention/query/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:09multi_head_self_attention/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ъ
1multi_head_self_attention/query/Tensordot/ReshapeReshape7multi_head_self_attention/query/Tensordot/transpose:y:08multi_head_self_attention/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ъ
0multi_head_self_attention/query/Tensordot/MatMulMatMul:multi_head_self_attention/query/Tensordot/Reshape:output:0@multi_head_self_attention/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @{
1multi_head_self_attention/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@y
7multi_head_self_attention/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
2multi_head_self_attention/query/Tensordot/concat_1ConcatV2;multi_head_self_attention/query/Tensordot/GatherV2:output:0:multi_head_self_attention/query/Tensordot/Const_2:output:0@multi_head_self_attention/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:у
)multi_head_self_attention/query/TensordotReshape:multi_head_self_attention/query/Tensordot/MatMul:product:0;multi_head_self_attention/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@▓
6multi_head_self_attention/query/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_query_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▄
'multi_head_self_attention/query/BiasAddBiasAdd2multi_head_self_attention/query/Tensordot:output:0>multi_head_self_attention/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@╢
6multi_head_self_attention/key/Tensordot/ReadVariableOpReadVariableOp?multi_head_self_attention_key_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0v
,multi_head_self_attention/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,multi_head_self_attention/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Д
-multi_head_self_attention/key/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:w
5multi_head_self_attention/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : │
0multi_head_self_attention/key/Tensordot/GatherV2GatherV26multi_head_self_attention/key/Tensordot/Shape:output:05multi_head_self_attention/key/Tensordot/free:output:0>multi_head_self_attention/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7multi_head_self_attention/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╖
2multi_head_self_attention/key/Tensordot/GatherV2_1GatherV26multi_head_self_attention/key/Tensordot/Shape:output:05multi_head_self_attention/key/Tensordot/axes:output:0@multi_head_self_attention/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-multi_head_self_attention/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╚
,multi_head_self_attention/key/Tensordot/ProdProd9multi_head_self_attention/key/Tensordot/GatherV2:output:06multi_head_self_attention/key/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/multi_head_self_attention/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/key/Tensordot/Prod_1Prod;multi_head_self_attention/key/Tensordot/GatherV2_1:output:08multi_head_self_attention/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3multi_head_self_attention/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.multi_head_self_attention/key/Tensordot/concatConcatV25multi_head_self_attention/key/Tensordot/free:output:05multi_head_self_attention/key/Tensordot/axes:output:0<multi_head_self_attention/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╙
-multi_head_self_attention/key/Tensordot/stackPack5multi_head_self_attention/key/Tensordot/Prod:output:07multi_head_self_attention/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╓
1multi_head_self_attention/key/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:07multi_head_self_attention/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ф
/multi_head_self_attention/key/Tensordot/ReshapeReshape5multi_head_self_attention/key/Tensordot/transpose:y:06multi_head_self_attention/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ф
.multi_head_self_attention/key/Tensordot/MatMulMatMul8multi_head_self_attention/key/Tensordot/Reshape:output:0>multi_head_self_attention/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
/multi_head_self_attention/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@w
5multi_head_self_attention/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0multi_head_self_attention/key/Tensordot/concat_1ConcatV29multi_head_self_attention/key/Tensordot/GatherV2:output:08multi_head_self_attention/key/Tensordot/Const_2:output:0>multi_head_self_attention/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:▌
'multi_head_self_attention/key/TensordotReshape8multi_head_self_attention/key/Tensordot/MatMul:product:09multi_head_self_attention/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@о
4multi_head_self_attention/key/BiasAdd/ReadVariableOpReadVariableOp=multi_head_self_attention_key_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╓
%multi_head_self_attention/key/BiasAddBiasAdd0multi_head_self_attention/key/Tensordot:output:0<multi_head_self_attention/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@║
8multi_head_self_attention/value/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_value_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0x
.multi_head_self_attention/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
.multi_head_self_attention/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ж
/multi_head_self_attention/value/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:y
7multi_head_self_attention/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
2multi_head_self_attention/value/Tensordot/GatherV2GatherV28multi_head_self_attention/value/Tensordot/Shape:output:07multi_head_self_attention/value/Tensordot/free:output:0@multi_head_self_attention/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
9multi_head_self_attention/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
4multi_head_self_attention/value/Tensordot/GatherV2_1GatherV28multi_head_self_attention/value/Tensordot/Shape:output:07multi_head_self_attention/value/Tensordot/axes:output:0Bmulti_head_self_attention/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
/multi_head_self_attention/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/value/Tensordot/ProdProd;multi_head_self_attention/value/Tensordot/GatherV2:output:08multi_head_self_attention/value/Tensordot/Const:output:0*
T0*
_output_shapes
: {
1multi_head_self_attention/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╘
0multi_head_self_attention/value/Tensordot/Prod_1Prod=multi_head_self_attention/value/Tensordot/GatherV2_1:output:0:multi_head_self_attention/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: w
5multi_head_self_attention/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
0multi_head_self_attention/value/Tensordot/concatConcatV27multi_head_self_attention/value/Tensordot/free:output:07multi_head_self_attention/value/Tensordot/axes:output:0>multi_head_self_attention/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┘
/multi_head_self_attention/value/Tensordot/stackPack7multi_head_self_attention/value/Tensordot/Prod:output:09multi_head_self_attention/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┌
3multi_head_self_attention/value/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:09multi_head_self_attention/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@ъ
1multi_head_self_attention/value/Tensordot/ReshapeReshape7multi_head_self_attention/value/Tensordot/transpose:y:08multi_head_self_attention/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ъ
0multi_head_self_attention/value/Tensordot/MatMulMatMul:multi_head_self_attention/value/Tensordot/Reshape:output:0@multi_head_self_attention/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @{
1multi_head_self_attention/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@y
7multi_head_self_attention/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
2multi_head_self_attention/value/Tensordot/concat_1ConcatV2;multi_head_self_attention/value/Tensordot/GatherV2:output:0:multi_head_self_attention/value/Tensordot/Const_2:output:0@multi_head_self_attention/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:у
)multi_head_self_attention/value/TensordotReshape:multi_head_self_attention/value/Tensordot/MatMul:product:0;multi_head_self_attention/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@▓
6multi_head_self_attention/value/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_value_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▄
'multi_head_self_attention/value/BiasAddBiasAdd2multi_head_self_attention/value/Tensordot:output:0>multi_head_self_attention/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@t
)multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         k
)multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :k
)multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :л
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:╙
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/query/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"                  Б
(multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╥
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :│
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:╒
#multi_head_self_attention/Reshape_1Reshape.multi_head_self_attention/key/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╪
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :m
+multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :│
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:╫
#multi_head_self_attention/Reshape_2Reshape0multi_head_self_attention/value/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╪
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"                  ▐
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+                           *
adj_y(z
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:В
/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         {
1multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▌
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskК
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: k
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: ╟
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+                           Я
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+                           ╬
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"                  Г
*multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╫
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"                  v
+multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         m
+multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@¤
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:╠
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :                  @╢
6multi_head_self_attention/out/Tensordot/ReadVariableOpReadVariableOp?multi_head_self_attention_out_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0v
,multi_head_self_attention/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:}
,multi_head_self_attention/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Й
-multi_head_self_attention/out/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:w
5multi_head_self_attention/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : │
0multi_head_self_attention/out/Tensordot/GatherV2GatherV26multi_head_self_attention/out/Tensordot/Shape:output:05multi_head_self_attention/out/Tensordot/free:output:0>multi_head_self_attention/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:y
7multi_head_self_attention/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╖
2multi_head_self_attention/out/Tensordot/GatherV2_1GatherV26multi_head_self_attention/out/Tensordot/Shape:output:05multi_head_self_attention/out/Tensordot/axes:output:0@multi_head_self_attention/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:w
-multi_head_self_attention/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╚
,multi_head_self_attention/out/Tensordot/ProdProd9multi_head_self_attention/out/Tensordot/GatherV2:output:06multi_head_self_attention/out/Tensordot/Const:output:0*
T0*
_output_shapes
: y
/multi_head_self_attention/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╬
.multi_head_self_attention/out/Tensordot/Prod_1Prod;multi_head_self_attention/out/Tensordot/GatherV2_1:output:08multi_head_self_attention/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: u
3multi_head_self_attention/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ф
.multi_head_self_attention/out/Tensordot/concatConcatV25multi_head_self_attention/out/Tensordot/free:output:05multi_head_self_attention/out/Tensordot/axes:output:0<multi_head_self_attention/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╙
-multi_head_self_attention/out/Tensordot/stackPack5multi_head_self_attention/out/Tensordot/Prod:output:07multi_head_self_attention/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ф
1multi_head_self_attention/out/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:07multi_head_self_attention/out/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @ф
/multi_head_self_attention/out/Tensordot/ReshapeReshape5multi_head_self_attention/out/Tensordot/transpose:y:06multi_head_self_attention/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ф
.multi_head_self_attention/out/Tensordot/MatMulMatMul8multi_head_self_attention/out/Tensordot/Reshape:output:0>multi_head_self_attention/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @y
/multi_head_self_attention/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@w
5multi_head_self_attention/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
0multi_head_self_attention/out/Tensordot/concat_1ConcatV29multi_head_self_attention/out/Tensordot/GatherV2:output:08multi_head_self_attention/out/Tensordot/Const_2:output:0>multi_head_self_attention/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ц
'multi_head_self_attention/out/TensordotReshape8multi_head_self_attention/out/Tensordot/MatMul:product:09multi_head_self_attention/out/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @о
4multi_head_self_attention/out/BiasAdd/ReadVariableOpReadVariableOp=multi_head_self_attention_out_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▀
%multi_head_self_attention/out/BiasAddBiasAdd0multi_head_self_attention/out/Tensordot:output:0<multi_head_self_attention/out/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?н
dropout_2/dropout/MulMul.multi_head_self_attention/out/BiasAdd:output:0 dropout_2/dropout/Const:output:0*
T0*4
_output_shapes"
 :                  @u
dropout_2/dropout/ShapeShape.multi_head_self_attention/out/BiasAdd:output:0*
T0*
_output_shapes
:н
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*4
_output_shapes"
 :                  @*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╤
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  @Р
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :                  @Ф
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*4
_output_shapes"
 :                  @g
addAddV2dropout_2/dropout/Mul_1:z:0inputs*
T0*+
_output_shapes
:         A@~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╣
"layer_normalization_1/moments/meanMeanadd:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(Э
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         A╕
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@В
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:э
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5├
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         AН
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         Aк
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╟
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@Ф
%layer_normalization_1/batchnorm/mul_1Muladd:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@╕
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@в
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0├
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@╕
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@д
-sequential/dense_mlp/Tensordot/ReadVariableOpReadVariableOp6sequential_dense_mlp_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0m
#sequential/dense_mlp/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential/dense_mlp/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential/dense_mlp/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:n
,sequential/dense_mlp/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'sequential/dense_mlp/Tensordot/GatherV2GatherV2-sequential/dense_mlp/Tensordot/Shape:output:0,sequential/dense_mlp/Tensordot/free:output:05sequential/dense_mlp/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential/dense_mlp/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)sequential/dense_mlp/Tensordot/GatherV2_1GatherV2-sequential/dense_mlp/Tensordot/Shape:output:0,sequential/dense_mlp/Tensordot/axes:output:07sequential/dense_mlp/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential/dense_mlp/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
#sequential/dense_mlp/Tensordot/ProdProd0sequential/dense_mlp/Tensordot/GatherV2:output:0-sequential/dense_mlp/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential/dense_mlp/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%sequential/dense_mlp/Tensordot/Prod_1Prod2sequential/dense_mlp/Tensordot/GatherV2_1:output:0/sequential/dense_mlp/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential/dense_mlp/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%sequential/dense_mlp/Tensordot/concatConcatV2,sequential/dense_mlp/Tensordot/free:output:0,sequential/dense_mlp/Tensordot/axes:output:03sequential/dense_mlp/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$sequential/dense_mlp/Tensordot/stackPack,sequential/dense_mlp/Tensordot/Prod:output:0.sequential/dense_mlp/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╞
(sequential/dense_mlp/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0.sequential/dense_mlp/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@╔
&sequential/dense_mlp/Tensordot/ReshapeReshape,sequential/dense_mlp/Tensordot/transpose:y:0-sequential/dense_mlp/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%sequential/dense_mlp/Tensordot/MatMulMatMul/sequential/dense_mlp/Tensordot/Reshape:output:05sequential/dense_mlp/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          p
&sequential/dense_mlp/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: n
,sequential/dense_mlp/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'sequential/dense_mlp/Tensordot/concat_1ConcatV20sequential/dense_mlp/Tensordot/GatherV2:output:0/sequential/dense_mlp/Tensordot/Const_2:output:05sequential/dense_mlp/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:┬
sequential/dense_mlp/TensordotReshape/sequential/dense_mlp/Tensordot/MatMul:product:00sequential/dense_mlp/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A Ь
+sequential/dense_mlp/BiasAdd/ReadVariableOpReadVariableOp4sequential_dense_mlp_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╗
sequential/dense_mlp/BiasAddBiasAdd'sequential/dense_mlp/Tensordot:output:03sequential/dense_mlp/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A d
sequential/dense_mlp/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?л
sequential/dense_mlp/Gelu/mulMul(sequential/dense_mlp/Gelu/mul/x:output:0%sequential/dense_mlp/BiasAdd:output:0*
T0*+
_output_shapes
:         A e
 sequential/dense_mlp/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?┤
!sequential/dense_mlp/Gelu/truedivRealDiv%sequential/dense_mlp/BiasAdd:output:0)sequential/dense_mlp/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A Б
sequential/dense_mlp/Gelu/ErfErf%sequential/dense_mlp/Gelu/truediv:z:0*
T0*+
_output_shapes
:         A d
sequential/dense_mlp/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?й
sequential/dense_mlp/Gelu/addAddV2(sequential/dense_mlp/Gelu/add/x:output:0!sequential/dense_mlp/Gelu/Erf:y:0*
T0*+
_output_shapes
:         A в
sequential/dense_mlp/Gelu/mul_1Mul!sequential/dense_mlp/Gelu/mul:z:0!sequential/dense_mlp/Gelu/add:z:0*
T0*+
_output_shapes
:         A e
 sequential/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?л
sequential/dropout/dropout/MulMul#sequential/dense_mlp/Gelu/mul_1:z:0)sequential/dropout/dropout/Const:output:0*
T0*+
_output_shapes
:         A s
 sequential/dropout/dropout/ShapeShape#sequential/dense_mlp/Gelu/mul_1:z:0*
T0*
_output_shapes
:╢
7sequential/dropout/dropout/random_uniform/RandomUniformRandomUniform)sequential/dropout/dropout/Shape:output:0*
T0*+
_output_shapes
:         A *
dtype0n
)sequential/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>у
'sequential/dropout/dropout/GreaterEqualGreaterEqual@sequential/dropout/dropout/random_uniform/RandomUniform:output:02sequential/dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A Щ
sequential/dropout/dropout/CastCast+sequential/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A ж
 sequential/dropout/dropout/Mul_1Mul"sequential/dropout/dropout/Mul:z:0#sequential/dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:         A ░
3sequential/dense_embed_dim/Tensordot/ReadVariableOpReadVariableOp<sequential_dense_embed_dim_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0s
)sequential/dense_embed_dim/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
)sequential/dense_embed_dim/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
*sequential/dense_embed_dim/Tensordot/ShapeShape$sequential/dropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:t
2sequential/dense_embed_dim/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : з
-sequential/dense_embed_dim/Tensordot/GatherV2GatherV23sequential/dense_embed_dim/Tensordot/Shape:output:02sequential/dense_embed_dim/Tensordot/free:output:0;sequential/dense_embed_dim/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:v
4sequential/dense_embed_dim/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : л
/sequential/dense_embed_dim/Tensordot/GatherV2_1GatherV23sequential/dense_embed_dim/Tensordot/Shape:output:02sequential/dense_embed_dim/Tensordot/axes:output:0=sequential/dense_embed_dim/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
*sequential/dense_embed_dim/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ┐
)sequential/dense_embed_dim/Tensordot/ProdProd6sequential/dense_embed_dim/Tensordot/GatherV2:output:03sequential/dense_embed_dim/Tensordot/Const:output:0*
T0*
_output_shapes
: v
,sequential/dense_embed_dim/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┼
+sequential/dense_embed_dim/Tensordot/Prod_1Prod8sequential/dense_embed_dim/Tensordot/GatherV2_1:output:05sequential/dense_embed_dim/Tensordot/Const_1:output:0*
T0*
_output_shapes
: r
0sequential/dense_embed_dim/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : И
+sequential/dense_embed_dim/Tensordot/concatConcatV22sequential/dense_embed_dim/Tensordot/free:output:02sequential/dense_embed_dim/Tensordot/axes:output:09sequential/dense_embed_dim/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╩
*sequential/dense_embed_dim/Tensordot/stackPack2sequential/dense_embed_dim/Tensordot/Prod:output:04sequential/dense_embed_dim/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:═
.sequential/dense_embed_dim/Tensordot/transpose	Transpose$sequential/dropout/dropout/Mul_1:z:04sequential/dense_embed_dim/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A █
,sequential/dense_embed_dim/Tensordot/ReshapeReshape2sequential/dense_embed_dim/Tensordot/transpose:y:03sequential/dense_embed_dim/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  █
+sequential/dense_embed_dim/Tensordot/MatMulMatMul5sequential/dense_embed_dim/Tensordot/Reshape:output:0;sequential/dense_embed_dim/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @v
,sequential/dense_embed_dim/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@t
2sequential/dense_embed_dim/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
-sequential/dense_embed_dim/Tensordot/concat_1ConcatV26sequential/dense_embed_dim/Tensordot/GatherV2:output:05sequential/dense_embed_dim/Tensordot/Const_2:output:0;sequential/dense_embed_dim/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╘
$sequential/dense_embed_dim/TensordotReshape5sequential/dense_embed_dim/Tensordot/MatMul:product:06sequential/dense_embed_dim/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@и
1sequential/dense_embed_dim/BiasAdd/ReadVariableOpReadVariableOp:sequential_dense_embed_dim_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0═
"sequential/dense_embed_dim/BiasAddBiasAdd-sequential/dense_embed_dim/Tensordot:output:09sequential/dense_embed_dim/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@g
"sequential/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?╖
 sequential/dropout_1/dropout/MulMul+sequential/dense_embed_dim/BiasAdd:output:0+sequential/dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:         A@}
"sequential/dropout_1/dropout/ShapeShape+sequential/dense_embed_dim/BiasAdd:output:0*
T0*
_output_shapes
:║
9sequential/dropout_1/dropout/random_uniform/RandomUniformRandomUniform+sequential/dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:         A@*
dtype0p
+sequential/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>щ
)sequential/dropout_1/dropout/GreaterEqualGreaterEqualBsequential/dropout_1/dropout/random_uniform/RandomUniform:output:04sequential/dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A@Э
!sequential/dropout_1/dropout/CastCast-sequential/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A@м
"sequential/dropout_1/dropout/Mul_1Mul$sequential/dropout_1/dropout/Mul:z:0%sequential/dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:         A@\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?Ь
dropout_3/dropout/MulMul&sequential/dropout_1/dropout/Mul_1:z:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:         A@m
dropout_3/dropout/ShapeShape&sequential/dropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:д
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:         A@*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>╚
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A@З
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A@Л
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:         A@j
add_1AddV2dropout_3/dropout/Mul_1:z:0add:z:0*
T0*+
_output_shapes
:         A@\
IdentityIdentity	add_1:z:0^NoOp*
T0*+
_output_shapes
:         A@Ю
NoOpNoOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp5^multi_head_self_attention/key/BiasAdd/ReadVariableOp7^multi_head_self_attention/key/Tensordot/ReadVariableOp5^multi_head_self_attention/out/BiasAdd/ReadVariableOp7^multi_head_self_attention/out/Tensordot/ReadVariableOp7^multi_head_self_attention/query/BiasAdd/ReadVariableOp9^multi_head_self_attention/query/Tensordot/ReadVariableOp7^multi_head_self_attention/value/BiasAdd/ReadVariableOp9^multi_head_self_attention/value/Tensordot/ReadVariableOp2^sequential/dense_embed_dim/BiasAdd/ReadVariableOp4^sequential/dense_embed_dim/Tensordot/ReadVariableOp,^sequential/dense_mlp/BiasAdd/ReadVariableOp.^sequential/dense_mlp/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         A@: : : : : : : : : : : : : : : : 2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2l
4multi_head_self_attention/key/BiasAdd/ReadVariableOp4multi_head_self_attention/key/BiasAdd/ReadVariableOp2p
6multi_head_self_attention/key/Tensordot/ReadVariableOp6multi_head_self_attention/key/Tensordot/ReadVariableOp2l
4multi_head_self_attention/out/BiasAdd/ReadVariableOp4multi_head_self_attention/out/BiasAdd/ReadVariableOp2p
6multi_head_self_attention/out/Tensordot/ReadVariableOp6multi_head_self_attention/out/Tensordot/ReadVariableOp2p
6multi_head_self_attention/query/BiasAdd/ReadVariableOp6multi_head_self_attention/query/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/query/Tensordot/ReadVariableOp8multi_head_self_attention/query/Tensordot/ReadVariableOp2p
6multi_head_self_attention/value/BiasAdd/ReadVariableOp6multi_head_self_attention/value/BiasAdd/ReadVariableOp2t
8multi_head_self_attention/value/Tensordot/ReadVariableOp8multi_head_self_attention/value/Tensordot/ReadVariableOp2f
1sequential/dense_embed_dim/BiasAdd/ReadVariableOp1sequential/dense_embed_dim/BiasAdd/ReadVariableOp2j
3sequential/dense_embed_dim/Tensordot/ReadVariableOp3sequential/dense_embed_dim/Tensordot/ReadVariableOp2Z
+sequential/dense_mlp/BiasAdd/ReadVariableOp+sequential/dense_mlp/BiasAdd/ReadVariableOp2^
-sequential/dense_mlp/Tensordot/ReadVariableOp-sequential/dense_mlp/Tensordot/ReadVariableOp:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
м
C
'__inference_dropout_layer_call_fn_38144

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_34405d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         A "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A :S O
+
_output_shapes
:         A 
 
_user_specified_nameinputs
Р

a
B__inference_dropout_layer_call_and_return_conditional_losses_34515

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         A C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         A *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         A ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         A "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A :S O
+
_output_shapes
:         A 
 
_user_specified_nameinputs
╡A
Ц
__inference__traced_save_38345
file_prefix&
"savev2_pos_emb_read_readvariableop(
$savev2_class_emb_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_query_kernel_read_readvariableopU
Qsavev2_transformer_block_multi_head_self_attention_query_bias_read_readvariableopU
Qsavev2_transformer_block_multi_head_self_attention_key_kernel_read_readvariableopS
Osavev2_transformer_block_multi_head_self_attention_key_bias_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_value_kernel_read_readvariableopU
Qsavev2_transformer_block_multi_head_self_attention_value_bias_read_readvariableopU
Qsavev2_transformer_block_multi_head_self_attention_out_kernel_read_readvariableopS
Osavev2_transformer_block_multi_head_self_attention_out_bias_read_readvariableop/
+savev2_dense_mlp_kernel_read_readvariableop-
)savev2_dense_mlp_bias_read_readvariableop5
1savev2_dense_embed_dim_kernel_read_readvariableop3
/savev2_dense_embed_dim_bias_read_readvariableopJ
Fsavev2_transformer_block_layer_normalization_gamma_read_readvariableopI
Esavev2_transformer_block_layer_normalization_beta_read_readvariableopL
Hsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopK
Gsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ·

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*г

valueЩ
BЦ
B"pos_emb/.ATTRIBUTES/VARIABLE_VALUEB$class_emb/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHл
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Б
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0"savev2_pos_emb_read_readvariableop$savev2_class_emb_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_query_kernel_read_readvariableopQsavev2_transformer_block_multi_head_self_attention_query_bias_read_readvariableopQsavev2_transformer_block_multi_head_self_attention_key_kernel_read_readvariableopOsavev2_transformer_block_multi_head_self_attention_key_bias_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_value_kernel_read_readvariableopQsavev2_transformer_block_multi_head_self_attention_value_bias_read_readvariableopQsavev2_transformer_block_multi_head_self_attention_out_kernel_read_readvariableopOsavev2_transformer_block_multi_head_self_attention_out_bias_read_readvariableop+savev2_dense_mlp_kernel_read_readvariableop)savev2_dense_mlp_bias_read_readvariableop1savev2_dense_embed_dim_kernel_read_readvariableop/savev2_dense_embed_dim_bias_read_readvariableopFsavev2_transformer_block_layer_normalization_gamma_read_readvariableopEsavev2_transformer_block_layer_normalization_beta_read_readvariableopHsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopGsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ё
_input_shapes▀
▄: :A@:@:@:@:@@:@:@@:@:@@:@:@@:@:@ : : @:@:@:@:@:@:@:@:@ : : 
:
: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:A@:($
"
_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: 
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
х
`
B__inference_dropout_layer_call_and_return_conditional_losses_34405

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:         A _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         A "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A :S O
+
_output_shapes
:         A 
 
_user_specified_nameinputs
Є	
c
D__inference_dropout_4_layer_call_and_return_conditional_losses_37901

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┴
Ф
'__inference_dense_2_layer_call_fn_37910

inputs
unknown: 

	unknown_0:

identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_34691o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
я
Б
,__inference_sequential_1_layer_call_fn_37115

inputs
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 

	unknown_4:

identityИвStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_34805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┼	
є
B__inference_dense_2_layer_call_and_return_conditional_losses_34691

inputs0
matmul_readvariableop_resource: 
-
biasadd_readvariableop_resource:

identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┘
Ю
5__inference_layer_normalization_2_layer_call_fn_37825

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_34644o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╝
и
#__inference_signature_wrapper_36176
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:A@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16: 

unknown_17: @

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@ 

unknown_22: 

unknown_23: 


unknown_24:

identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_34349o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_1
э║
╠%
 __inference__wrapped_model_34349
input_1L
:vision_transformer_dense_tensordot_readvariableop_resource:@F
8vision_transformer_dense_biasadd_readvariableop_resource:@L
6vision_transformer_broadcastto_readvariableop_resource:@D
.vision_transformer_add_readvariableop_resource:A@l
^vision_transformer_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource:@h
Zvision_transformer_transformer_block_layer_normalization_batchnorm_readvariableop_resource:@x
fvision_transformer_transformer_block_multi_head_self_attention_query_tensordot_readvariableop_resource:@@r
dvision_transformer_transformer_block_multi_head_self_attention_query_biasadd_readvariableop_resource:@v
dvision_transformer_transformer_block_multi_head_self_attention_key_tensordot_readvariableop_resource:@@p
bvision_transformer_transformer_block_multi_head_self_attention_key_biasadd_readvariableop_resource:@x
fvision_transformer_transformer_block_multi_head_self_attention_value_tensordot_readvariableop_resource:@@r
dvision_transformer_transformer_block_multi_head_self_attention_value_biasadd_readvariableop_resource:@v
dvision_transformer_transformer_block_multi_head_self_attention_out_tensordot_readvariableop_resource:@@p
bvision_transformer_transformer_block_multi_head_self_attention_out_biasadd_readvariableop_resource:@n
`vision_transformer_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource:@j
\vision_transformer_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource:@m
[vision_transformer_transformer_block_sequential_dense_mlp_tensordot_readvariableop_resource:@ g
Yvision_transformer_transformer_block_sequential_dense_mlp_biasadd_readvariableop_resource: s
avision_transformer_transformer_block_sequential_dense_embed_dim_tensordot_readvariableop_resource: @m
_vision_transformer_transformer_block_sequential_dense_embed_dim_biasadd_readvariableop_resource:@i
[vision_transformer_sequential_1_layer_normalization_2_batchnorm_mul_readvariableop_resource:@e
Wvision_transformer_sequential_1_layer_normalization_2_batchnorm_readvariableop_resource:@X
Fvision_transformer_sequential_1_dense_1_matmul_readvariableop_resource:@ U
Gvision_transformer_sequential_1_dense_1_biasadd_readvariableop_resource: X
Fvision_transformer_sequential_1_dense_2_matmul_readvariableop_resource: 
U
Gvision_transformer_sequential_1_dense_2_biasadd_readvariableop_resource:

identityИв-vision_transformer/BroadcastTo/ReadVariableOpв%vision_transformer/add/ReadVariableOpв/vision_transformer/dense/BiasAdd/ReadVariableOpв1vision_transformer/dense/Tensordot/ReadVariableOpв>vision_transformer/sequential_1/dense_1/BiasAdd/ReadVariableOpв=vision_transformer/sequential_1/dense_1/MatMul/ReadVariableOpв>vision_transformer/sequential_1/dense_2/BiasAdd/ReadVariableOpв=vision_transformer/sequential_1/dense_2/MatMul/ReadVariableOpвNvision_transformer/sequential_1/layer_normalization_2/batchnorm/ReadVariableOpвRvision_transformer/sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOpвQvision_transformer/transformer_block/layer_normalization/batchnorm/ReadVariableOpвUvision_transformer/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpвSvision_transformer/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpвWvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpвYvision_transformer/transformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpв[vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpвYvision_transformer/transformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpв[vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpв[vision_transformer/transformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpв]vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpв[vision_transformer/transformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpв]vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpвVvision_transformer/transformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpвXvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpвPvision_transformer/transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOpвRvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOpO
vision_transformer/ShapeShapeinput_1*
T0*
_output_shapes
:p
&vision_transformer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(vision_transformer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(vision_transformer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 vision_transformer/strided_sliceStridedSlice!vision_transformer/Shape:output:0/vision_transformer/strided_slice/stack:output:01vision_transformer/strided_slice/stack_1:output:01vision_transformer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
#vision_transformer/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;j
%vision_transformer/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
 vision_transformer/rescaling/mulMulinput_1,vision_transformer/rescaling/Cast/x:output:0*
T0*/
_output_shapes
:           ╣
 vision_transformer/rescaling/addAddV2$vision_transformer/rescaling/mul:z:0.vision_transformer/rescaling/Cast_1/x:output:0*
T0*/
_output_shapes
:           n
vision_transformer/Shape_1Shape$vision_transformer/rescaling/add:z:0*
T0*
_output_shapes
:r
(vision_transformer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*vision_transformer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*vision_transformer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
"vision_transformer/strided_slice_1StridedSlice#vision_transformer/Shape_1:output:01vision_transformer/strided_slice_1/stack:output:03vision_transformer/strided_slice_1/stack_1:output:03vision_transformer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskэ
&vision_transformer/ExtractImagePatchesExtractImagePatches$vision_transformer/rescaling/add:z:0*
T0*/
_output_shapes
:         *
ksizes
*
paddingVALID*
rates
*
strides
m
"vision_transformer/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         d
"vision_transformer/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :▌
 vision_transformer/Reshape/shapePack+vision_transformer/strided_slice_1:output:0+vision_transformer/Reshape/shape/1:output:0+vision_transformer/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:┴
vision_transformer/ReshapeReshape0vision_transformer/ExtractImagePatches:patches:0)vision_transformer/Reshape/shape:output:0*
T0*4
_output_shapes"
 :                  м
1vision_transformer/dense/Tensordot/ReadVariableOpReadVariableOp:vision_transformer_dense_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0q
'vision_transformer/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:x
'vision_transformer/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
(vision_transformer/dense/Tensordot/ShapeShape#vision_transformer/Reshape:output:0*
T0*
_output_shapes
:r
0vision_transformer/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
+vision_transformer/dense/Tensordot/GatherV2GatherV21vision_transformer/dense/Tensordot/Shape:output:00vision_transformer/dense/Tensordot/free:output:09vision_transformer/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:t
2vision_transformer/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
-vision_transformer/dense/Tensordot/GatherV2_1GatherV21vision_transformer/dense/Tensordot/Shape:output:00vision_transformer/dense/Tensordot/axes:output:0;vision_transformer/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
(vision_transformer/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╣
'vision_transformer/dense/Tensordot/ProdProd4vision_transformer/dense/Tensordot/GatherV2:output:01vision_transformer/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: t
*vision_transformer/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┐
)vision_transformer/dense/Tensordot/Prod_1Prod6vision_transformer/dense/Tensordot/GatherV2_1:output:03vision_transformer/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: p
.vision_transformer/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : А
)vision_transformer/dense/Tensordot/concatConcatV20vision_transformer/dense/Tensordot/free:output:00vision_transformer/dense/Tensordot/axes:output:07vision_transformer/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:─
(vision_transformer/dense/Tensordot/stackPack0vision_transformer/dense/Tensordot/Prod:output:02vision_transformer/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╤
,vision_transformer/dense/Tensordot/transpose	Transpose#vision_transformer/Reshape:output:02vision_transformer/dense/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  ╒
*vision_transformer/dense/Tensordot/ReshapeReshape0vision_transformer/dense/Tensordot/transpose:y:01vision_transformer/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╒
)vision_transformer/dense/Tensordot/MatMulMatMul3vision_transformer/dense/Tensordot/Reshape:output:09vision_transformer/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @t
*vision_transformer/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@r
0vision_transformer/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
+vision_transformer/dense/Tensordot/concat_1ConcatV24vision_transformer/dense/Tensordot/GatherV2:output:03vision_transformer/dense/Tensordot/Const_2:output:09vision_transformer/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╫
"vision_transformer/dense/TensordotReshape3vision_transformer/dense/Tensordot/MatMul:product:04vision_transformer/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @д
/vision_transformer/dense/BiasAdd/ReadVariableOpReadVariableOp8vision_transformer_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╨
 vision_transformer/dense/BiasAddBiasAdd+vision_transformer/dense/Tensordot:output:07vision_transformer/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @и
-vision_transformer/BroadcastTo/ReadVariableOpReadVariableOp6vision_transformer_broadcastto_readvariableop_resource*"
_output_shapes
:@*
dtype0h
&vision_transformer/BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :h
&vision_transformer/BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@ч
$vision_transformer/BroadcastTo/shapePack)vision_transformer/strided_slice:output:0/vision_transformer/BroadcastTo/shape/1:output:0/vision_transformer/BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:╔
vision_transformer/BroadcastToBroadcastTo5vision_transformer/BroadcastTo/ReadVariableOp:value:0-vision_transformer/BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         @`
vision_transformer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ъ
vision_transformer/concatConcatV2'vision_transformer/BroadcastTo:output:0)vision_transformer/dense/BiasAdd:output:0'vision_transformer/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  @Ш
%vision_transformer/add/ReadVariableOpReadVariableOp.vision_transformer_add_readvariableop_resource*"
_output_shapes
:A@*
dtype0и
vision_transformer/addAddV2"vision_transformer/concat:output:0-vision_transformer/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@б
Wvision_transformer/transformer_block/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Т
Evision_transformer/transformer_block/layer_normalization/moments/meanMeanvision_transformer/add:z:0`vision_transformer/transformer_block/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(у
Mvision_transformer/transformer_block/layer_normalization/moments/StopGradientStopGradientNvision_transformer/transformer_block/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         AС
Rvision_transformer/transformer_block/layer_normalization/moments/SquaredDifferenceSquaredDifferencevision_transformer/add:z:0Vvision_transformer/transformer_block/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@е
[vision_transformer/transformer_block/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╓
Ivision_transformer/transformer_block/layer_normalization/moments/varianceMeanVvision_transformer/transformer_block/layer_normalization/moments/SquaredDifference:z:0dvision_transformer/transformer_block/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(Н
Hvision_transformer/transformer_block/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5м
Fvision_transformer/transformer_block/layer_normalization/batchnorm/addAddV2Rvision_transformer/transformer_block/layer_normalization/moments/variance:output:0Qvision_transformer/transformer_block/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         A╙
Hvision_transformer/transformer_block/layer_normalization/batchnorm/RsqrtRsqrtJvision_transformer/transformer_block/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         AЁ
Uvision_transformer/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp^vision_transformer_transformer_block_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0░
Fvision_transformer/transformer_block/layer_normalization/batchnorm/mulMulLvision_transformer/transformer_block/layer_normalization/batchnorm/Rsqrt:y:0]vision_transformer/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@э
Hvision_transformer/transformer_block/layer_normalization/batchnorm/mul_1Mulvision_transformer/add:z:0Jvision_transformer/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@б
Hvision_transformer/transformer_block/layer_normalization/batchnorm/mul_2MulNvision_transformer/transformer_block/layer_normalization/moments/mean:output:0Jvision_transformer/transformer_block/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@ш
Qvision_transformer/transformer_block/layer_normalization/batchnorm/ReadVariableOpReadVariableOpZvision_transformer_transformer_block_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0м
Fvision_transformer/transformer_block/layer_normalization/batchnorm/subSubYvision_transformer/transformer_block/layer_normalization/batchnorm/ReadVariableOp:value:0Lvision_transformer/transformer_block/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@б
Hvision_transformer/transformer_block/layer_normalization/batchnorm/add_1AddV2Lvision_transformer/transformer_block/layer_normalization/batchnorm/mul_1:z:0Jvision_transformer/transformer_block/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@└
Dvision_transformer/transformer_block/multi_head_self_attention/ShapeShapeLvision_transformer/transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Ь
Rvision_transformer/transformer_block/multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ю
Tvision_transformer/transformer_block/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ю
Tvision_transformer/transformer_block/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
Lvision_transformer/transformer_block/multi_head_self_attention/strided_sliceStridedSliceMvision_transformer/transformer_block/multi_head_self_attention/Shape:output:0[vision_transformer/transformer_block/multi_head_self_attention/strided_slice/stack:output:0]vision_transformer/transformer_block/multi_head_self_attention/strided_slice/stack_1:output:0]vision_transformer/transformer_block/multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskД
]vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOpReadVariableOpfvision_transformer_transformer_block_multi_head_self_attention_query_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0Э
Svision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:д
Svision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ╨
Tvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ShapeShapeLvision_transformer/transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Ю
\vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
Wvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/GatherV2GatherV2]vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Shape:output:0\vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/free:output:0evision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:а
^vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
Yvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1GatherV2]vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Shape:output:0\vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/axes:output:0gvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ю
Tvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╜
Svision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ProdProd`vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/GatherV2:output:0]vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Const:output:0*
T0*
_output_shapes
: а
Vvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ├
Uvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Prod_1Prodbvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/GatherV2_1:output:0_vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Ь
Zvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ░
Uvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/concatConcatV2\vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/free:output:0\vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/axes:output:0cvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╚
Tvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/stackPack\vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Prod:output:0^vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╔
Xvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/transpose	TransposeLvision_transformer/transformer_block/layer_normalization/batchnorm/add_1:z:0^vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@┘
Vvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ReshapeReshape\vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/transpose:y:0]vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ┘
Uvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/MatMulMatMul_vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Reshape:output:0evision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
Vvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Ю
\vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Wvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/concat_1ConcatV2`vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/GatherV2:output:0_vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/Const_2:output:0evision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╥
Nvision_transformer/transformer_block/multi_head_self_attention/query/TensordotReshape_vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/MatMul:product:0`vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@№
[vision_transformer/transformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOpReadVariableOpdvision_transformer_transformer_block_multi_head_self_attention_query_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╦
Lvision_transformer/transformer_block/multi_head_self_attention/query/BiasAddBiasAddWvision_transformer/transformer_block/multi_head_self_attention/query/Tensordot:output:0cvision_transformer/transformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@А
[vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpReadVariableOpdvision_transformer_transformer_block_multi_head_self_attention_key_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0Ы
Qvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:в
Qvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
Rvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ShapeShapeLvision_transformer/transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Ь
Zvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
Uvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/GatherV2GatherV2[vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Shape:output:0Zvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/free:output:0cvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ю
\vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
Wvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1GatherV2[vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Shape:output:0Zvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/axes:output:0evision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ь
Rvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╖
Qvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ProdProd^vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/GatherV2:output:0[vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Const:output:0*
T0*
_output_shapes
: Ю
Tvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╜
Svision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Prod_1Prod`vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/GatherV2_1:output:0]vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Ъ
Xvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : и
Svision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/concatConcatV2Zvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/free:output:0Zvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/axes:output:0avision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┬
Rvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/stackPackZvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Prod:output:0\vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┼
Vvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/transpose	TransposeLvision_transformer/transformer_block/layer_normalization/batchnorm/add_1:z:0\vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@╙
Tvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ReshapeReshapeZvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/transpose:y:0[vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╙
Svision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/MatMulMatMul]vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Reshape:output:0cvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
Tvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Ь
Zvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : │
Uvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/concat_1ConcatV2^vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/GatherV2:output:0]vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/Const_2:output:0cvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╠
Lvision_transformer/transformer_block/multi_head_self_attention/key/TensordotReshape]vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/MatMul:product:0^vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@°
Yvision_transformer/transformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpReadVariableOpbvision_transformer_transformer_block_multi_head_self_attention_key_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┼
Jvision_transformer/transformer_block/multi_head_self_attention/key/BiasAddBiasAddUvision_transformer/transformer_block/multi_head_self_attention/key/Tensordot:output:0avision_transformer/transformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@Д
]vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpReadVariableOpfvision_transformer_transformer_block_multi_head_self_attention_value_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0Э
Svision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:д
Svision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ╨
Tvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ShapeShapeLvision_transformer/transformer_block/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:Ю
\vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
Wvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/GatherV2GatherV2]vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Shape:output:0\vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/free:output:0evision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:а
^vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╙
Yvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1GatherV2]vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Shape:output:0\vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/axes:output:0gvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ю
Tvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╜
Svision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ProdProd`vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/GatherV2:output:0]vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Const:output:0*
T0*
_output_shapes
: а
Vvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ├
Uvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Prod_1Prodbvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/GatherV2_1:output:0_vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Ь
Zvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ░
Uvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/concatConcatV2\vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/free:output:0\vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/axes:output:0cvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╚
Tvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/stackPack\vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Prod:output:0^vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╔
Xvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/transpose	TransposeLvision_transformer/transformer_block/layer_normalization/batchnorm/add_1:z:0^vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@┘
Vvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ReshapeReshape\vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/transpose:y:0]vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ┘
Uvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/MatMulMatMul_vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Reshape:output:0evision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
Vvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Ю
\vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Wvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/concat_1ConcatV2`vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/GatherV2:output:0_vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/Const_2:output:0evision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╥
Nvision_transformer/transformer_block/multi_head_self_attention/value/TensordotReshape_vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/MatMul:product:0`vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@№
[vision_transformer/transformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOpReadVariableOpdvision_transformer_transformer_block_multi_head_self_attention_value_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╦
Lvision_transformer/transformer_block/multi_head_self_attention/value/BiasAddBiasAddWvision_transformer/transformer_block/multi_head_self_attention/value/Tensordot:output:0cvision_transformer/transformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@Щ
Nvision_transformer/transformer_block/multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Р
Nvision_transformer/transformer_block/multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Р
Nvision_transformer/transformer_block/multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ф
Lvision_transformer/transformer_block/multi_head_self_attention/Reshape/shapePackUvision_transformer/transformer_block/multi_head_self_attention/strided_slice:output:0Wvision_transformer/transformer_block/multi_head_self_attention/Reshape/shape/1:output:0Wvision_transformer/transformer_block/multi_head_self_attention/Reshape/shape/2:output:0Wvision_transformer/transformer_block/multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:┬
Fvision_transformer/transformer_block/multi_head_self_attention/ReshapeReshapeUvision_transformer/transformer_block/multi_head_self_attention/query/BiasAdd:output:0Uvision_transformer/transformer_block/multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"                  ж
Mvision_transformer/transformer_block/multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ┴
Hvision_transformer/transformer_block/multi_head_self_attention/transpose	TransposeOvision_transformer/transformer_block/multi_head_self_attention/Reshape:output:0Vvision_transformer/transformer_block/multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"                  Ы
Pvision_transformer/transformer_block/multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Т
Pvision_transformer/transformer_block/multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Т
Pvision_transformer/transformer_block/multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ь
Nvision_transformer/transformer_block/multi_head_self_attention/Reshape_1/shapePackUvision_transformer/transformer_block/multi_head_self_attention/strided_slice:output:0Yvision_transformer/transformer_block/multi_head_self_attention/Reshape_1/shape/1:output:0Yvision_transformer/transformer_block/multi_head_self_attention/Reshape_1/shape/2:output:0Yvision_transformer/transformer_block/multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:─
Hvision_transformer/transformer_block/multi_head_self_attention/Reshape_1ReshapeSvision_transformer/transformer_block/multi_head_self_attention/key/BiasAdd:output:0Wvision_transformer/transformer_block/multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"                  и
Ovision_transformer/transformer_block/multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╟
Jvision_transformer/transformer_block/multi_head_self_attention/transpose_1	TransposeQvision_transformer/transformer_block/multi_head_self_attention/Reshape_1:output:0Xvision_transformer/transformer_block/multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"                  Ы
Pvision_transformer/transformer_block/multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Т
Pvision_transformer/transformer_block/multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Т
Pvision_transformer/transformer_block/multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ь
Nvision_transformer/transformer_block/multi_head_self_attention/Reshape_2/shapePackUvision_transformer/transformer_block/multi_head_self_attention/strided_slice:output:0Yvision_transformer/transformer_block/multi_head_self_attention/Reshape_2/shape/1:output:0Yvision_transformer/transformer_block/multi_head_self_attention/Reshape_2/shape/2:output:0Yvision_transformer/transformer_block/multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:╞
Hvision_transformer/transformer_block/multi_head_self_attention/Reshape_2ReshapeUvision_transformer/transformer_block/multi_head_self_attention/value/BiasAdd:output:0Wvision_transformer/transformer_block/multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"                  и
Ovision_transformer/transformer_block/multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╟
Jvision_transformer/transformer_block/multi_head_self_attention/transpose_2	TransposeQvision_transformer/transformer_block/multi_head_self_attention/Reshape_2:output:0Xvision_transformer/transformer_block/multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"                  ═
Evision_transformer/transformer_block/multi_head_self_attention/MatMulBatchMatMulV2Lvision_transformer/transformer_block/multi_head_self_attention/transpose:y:0Nvision_transformer/transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+                           *
adj_y(─
Fvision_transformer/transformer_block/multi_head_self_attention/Shape_1ShapeNvision_transformer/transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:з
Tvision_transformer/transformer_block/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         а
Vvision_transformer/transformer_block/multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: а
Vvision_transformer/transformer_block/multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
Nvision_transformer/transformer_block/multi_head_self_attention/strided_slice_1StridedSliceOvision_transformer/transformer_block/multi_head_self_attention/Shape_1:output:0]vision_transformer/transformer_block/multi_head_self_attention/strided_slice_1/stack:output:0_vision_transformer/transformer_block/multi_head_self_attention/strided_slice_1/stack_1:output:0_vision_transformer/transformer_block/multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╘
Cvision_transformer/transformer_block/multi_head_self_attention/CastCastWvision_transformer/transformer_block/multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: ╡
Cvision_transformer/transformer_block/multi_head_self_attention/SqrtSqrtGvision_transformer/transformer_block/multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: ╢
Fvision_transformer/transformer_block/multi_head_self_attention/truedivRealDivNvision_transformer/transformer_block/multi_head_self_attention/MatMul:output:0Gvision_transformer/transformer_block/multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+                           щ
Fvision_transformer/transformer_block/multi_head_self_attention/SoftmaxSoftmaxJvision_transformer/transformer_block/multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+                           ╜
Gvision_transformer/transformer_block/multi_head_self_attention/MatMul_1BatchMatMulV2Pvision_transformer/transformer_block/multi_head_self_attention/Softmax:softmax:0Nvision_transformer/transformer_block/multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"                  и
Ovision_transformer/transformer_block/multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             ╞
Jvision_transformer/transformer_block/multi_head_self_attention/transpose_3	TransposePvision_transformer/transformer_block/multi_head_self_attention/MatMul_1:output:0Xvision_transformer/transformer_block/multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"                  Ы
Pvision_transformer/transformer_block/multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Т
Pvision_transformer/transformer_block/multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@С
Nvision_transformer/transformer_block/multi_head_self_attention/Reshape_3/shapePackUvision_transformer/transformer_block/multi_head_self_attention/strided_slice:output:0Yvision_transformer/transformer_block/multi_head_self_attention/Reshape_3/shape/1:output:0Yvision_transformer/transformer_block/multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:╗
Hvision_transformer/transformer_block/multi_head_self_attention/Reshape_3ReshapeNvision_transformer/transformer_block/multi_head_self_attention/transpose_3:y:0Wvision_transformer/transformer_block/multi_head_self_attention/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :                  @А
[vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOpReadVariableOpdvision_transformer_transformer_block_multi_head_self_attention_out_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0Ы
Qvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:в
Qvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ╙
Rvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ShapeShapeQvision_transformer/transformer_block/multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:Ь
Zvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
Uvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/GatherV2GatherV2[vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Shape:output:0Zvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/free:output:0cvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ю
\vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
Wvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1GatherV2[vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Shape:output:0Zvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/axes:output:0evision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ь
Rvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╖
Qvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ProdProd^vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/GatherV2:output:0[vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Const:output:0*
T0*
_output_shapes
: Ю
Tvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╜
Svision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Prod_1Prod`vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/GatherV2_1:output:0]vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Ъ
Xvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : и
Svision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/concatConcatV2Zvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/free:output:0Zvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/axes:output:0avision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:┬
Rvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/stackPackZvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Prod:output:0\vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╙
Vvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/transpose	TransposeQvision_transformer/transformer_block/multi_head_self_attention/Reshape_3:output:0\vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @╙
Tvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ReshapeReshapeZvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/transpose:y:0[vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╙
Svision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/MatMulMatMul]vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Reshape:output:0cvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ю
Tvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Ь
Zvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : │
Uvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/concat_1ConcatV2^vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/GatherV2:output:0]vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/Const_2:output:0cvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╒
Lvision_transformer/transformer_block/multi_head_self_attention/out/TensordotReshape]vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/MatMul:product:0^vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @°
Yvision_transformer/transformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpReadVariableOpbvision_transformer_transformer_block_multi_head_self_attention_out_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╬
Jvision_transformer/transformer_block/multi_head_self_attention/out/BiasAddBiasAddUvision_transformer/transformer_block/multi_head_self_attention/out/Tensordot:output:0avision_transformer/transformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @╫
7vision_transformer/transformer_block/dropout_2/IdentityIdentitySvision_transformer/transformer_block/multi_head_self_attention/out/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @┼
(vision_transformer/transformer_block/addAddV2@vision_transformer/transformer_block/dropout_2/Identity:output:0vision_transformer/add:z:0*
T0*+
_output_shapes
:         A@г
Yvision_transformer/transformer_block/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:и
Gvision_transformer/transformer_block/layer_normalization_1/moments/meanMean,vision_transformer/transformer_block/add:z:0bvision_transformer/transformer_block/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(ч
Ovision_transformer/transformer_block/layer_normalization_1/moments/StopGradientStopGradientPvision_transformer/transformer_block/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         Aз
Tvision_transformer/transformer_block/layer_normalization_1/moments/SquaredDifferenceSquaredDifference,vision_transformer/transformer_block/add:z:0Xvision_transformer/transformer_block/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         A@з
]vision_transformer/transformer_block/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:▄
Kvision_transformer/transformer_block/layer_normalization_1/moments/varianceMeanXvision_transformer/transformer_block/layer_normalization_1/moments/SquaredDifference:z:0fvision_transformer/transformer_block/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         A*
	keep_dims(П
Jvision_transformer/transformer_block/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5▓
Hvision_transformer/transformer_block/layer_normalization_1/batchnorm/addAddV2Tvision_transformer/transformer_block/layer_normalization_1/moments/variance:output:0Svision_transformer/transformer_block/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         A╫
Jvision_transformer/transformer_block/layer_normalization_1/batchnorm/RsqrtRsqrtLvision_transformer/transformer_block/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         AЇ
Wvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp`vision_transformer_transformer_block_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0╢
Hvision_transformer/transformer_block/layer_normalization_1/batchnorm/mulMulNvision_transformer/transformer_block/layer_normalization_1/batchnorm/Rsqrt:y:0_vision_transformer/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@Г
Jvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul_1Mul,vision_transformer/transformer_block/add:z:0Lvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@з
Jvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul_2MulPvision_transformer/transformer_block/layer_normalization_1/moments/mean:output:0Lvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         A@ь
Svision_transformer/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp\vision_transformer_transformer_block_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0▓
Hvision_transformer/transformer_block/layer_normalization_1/batchnorm/subSub[vision_transformer/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp:value:0Nvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         A@з
Jvision_transformer/transformer_block/layer_normalization_1/batchnorm/add_1AddV2Nvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul_1:z:0Lvision_transformer/transformer_block/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         A@ю
Rvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOpReadVariableOp[vision_transformer_transformer_block_sequential_dense_mlp_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype0Т
Hvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Щ
Hvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ╟
Ivision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ShapeShapeNvision_transformer/transformer_block/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:У
Qvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : г
Lvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/GatherV2GatherV2Rvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Shape:output:0Qvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/free:output:0Zvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Х
Svision_transformer/transformer_block/sequential/dense_mlp/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Nvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/GatherV2_1GatherV2Rvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Shape:output:0Qvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/axes:output:0\vision_transformer/transformer_block/sequential/dense_mlp/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:У
Ivision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ь
Hvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ProdProdUvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/GatherV2:output:0Rvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Const:output:0*
T0*
_output_shapes
: Х
Kvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: в
Jvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Prod_1ProdWvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/GatherV2_1:output:0Tvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Const_1:output:0*
T0*
_output_shapes
: С
Ovision_transformer/transformer_block/sequential/dense_mlp/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Jvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/concatConcatV2Qvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/free:output:0Qvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/axes:output:0Xvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:з
Ivision_transformer/transformer_block/sequential/dense_mlp/Tensordot/stackPackQvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Prod:output:0Svision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╡
Mvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/transpose	TransposeNvision_transformer/transformer_block/layer_normalization_1/batchnorm/add_1:z:0Svision_transformer/transformer_block/sequential/dense_mlp/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A@╕
Kvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ReshapeReshapeQvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/transpose:y:0Rvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╕
Jvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/MatMulMatMulTvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Reshape:output:0Zvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Х
Kvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: У
Qvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : П
Lvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/concat_1ConcatV2Uvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/GatherV2:output:0Tvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/Const_2:output:0Zvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:▒
Cvision_transformer/transformer_block/sequential/dense_mlp/TensordotReshapeTvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/MatMul:product:0Uvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A ц
Pvision_transformer/transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOpReadVariableOpYvision_transformer_transformer_block_sequential_dense_mlp_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0к
Avision_transformer/transformer_block/sequential/dense_mlp/BiasAddBiasAddLvision_transformer/transformer_block/sequential/dense_mlp/Tensordot:output:0Xvision_transformer/transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A Й
Dvision_transformer/transformer_block/sequential/dense_mlp/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ъ
Bvision_transformer/transformer_block/sequential/dense_mlp/Gelu/mulMulMvision_transformer/transformer_block/sequential/dense_mlp/Gelu/mul/x:output:0Jvision_transformer/transformer_block/sequential/dense_mlp/BiasAdd:output:0*
T0*+
_output_shapes
:         A К
Evision_transformer/transformer_block/sequential/dense_mlp/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?г
Fvision_transformer/transformer_block/sequential/dense_mlp/Gelu/truedivRealDivJvision_transformer/transformer_block/sequential/dense_mlp/BiasAdd:output:0Nvision_transformer/transformer_block/sequential/dense_mlp/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         A ╦
Bvision_transformer/transformer_block/sequential/dense_mlp/Gelu/ErfErfJvision_transformer/transformer_block/sequential/dense_mlp/Gelu/truediv:z:0*
T0*+
_output_shapes
:         A Й
Dvision_transformer/transformer_block/sequential/dense_mlp/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ш
Bvision_transformer/transformer_block/sequential/dense_mlp/Gelu/addAddV2Mvision_transformer/transformer_block/sequential/dense_mlp/Gelu/add/x:output:0Fvision_transformer/transformer_block/sequential/dense_mlp/Gelu/Erf:y:0*
T0*+
_output_shapes
:         A С
Dvision_transformer/transformer_block/sequential/dense_mlp/Gelu/mul_1MulFvision_transformer/transformer_block/sequential/dense_mlp/Gelu/mul:z:0Fvision_transformer/transformer_block/sequential/dense_mlp/Gelu/add:z:0*
T0*+
_output_shapes
:         A ╠
@vision_transformer/transformer_block/sequential/dropout/IdentityIdentityHvision_transformer/transformer_block/sequential/dense_mlp/Gelu/mul_1:z:0*
T0*+
_output_shapes
:         A ·
Xvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpReadVariableOpavision_transformer_transformer_block_sequential_dense_embed_dim_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype0Ш
Nvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Я
Nvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ╚
Ovision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ShapeShapeIvision_transformer/transformer_block/sequential/dropout/Identity:output:0*
T0*
_output_shapes
:Щ
Wvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Rvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2GatherV2Xvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Shape:output:0Wvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/free:output:0`vision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ы
Yvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1GatherV2Xvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Shape:output:0Wvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/axes:output:0bvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Щ
Ovision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: о
Nvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ProdProd[vision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2:output:0Xvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Const:output:0*
T0*
_output_shapes
: Ы
Qvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┤
Pvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Prod_1Prod]vision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2_1:output:0Zvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Ч
Uvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Pvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/concatConcatV2Wvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/free:output:0Wvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/axes:output:0^vision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╣
Ovision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/stackPackWvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Prod:output:0Yvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╝
Svision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/transpose	TransposeIvision_transformer/transformer_block/sequential/dropout/Identity:output:0Yvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/concat:output:0*
T0*+
_output_shapes
:         A ╩
Qvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ReshapeReshapeWvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/transpose:y:0Xvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╩
Pvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/MatMulMatMulZvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Reshape:output:0`vision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ы
Qvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Щ
Wvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Rvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/concat_1ConcatV2[vision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/GatherV2:output:0Zvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/Const_2:output:0`vision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:├
Ivision_transformer/transformer_block/sequential/dense_embed_dim/TensordotReshapeZvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/MatMul:product:0[vision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         A@Є
Vvision_transformer/transformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpReadVariableOp_vision_transformer_transformer_block_sequential_dense_embed_dim_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╝
Gvision_transformer/transformer_block/sequential/dense_embed_dim/BiasAddBiasAddRvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot:output:0^vision_transformer/transformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@╓
Bvision_transformer/transformer_block/sequential/dropout_1/IdentityIdentityPvision_transformer/transformer_block/sequential/dense_embed_dim/BiasAdd:output:0*
T0*+
_output_shapes
:         A@╞
7vision_transformer/transformer_block/dropout_3/IdentityIdentityKvision_transformer/transformer_block/sequential/dropout_1/Identity:output:0*
T0*+
_output_shapes
:         A@┘
*vision_transformer/transformer_block/add_1AddV2@vision_transformer/transformer_block/dropout_3/Identity:output:0,vision_transformer/transformer_block/add:z:0*
T0*+
_output_shapes
:         A@y
(vision_transformer/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*vision_transformer/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*vision_transformer/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      °
"vision_transformer/strided_slice_2StridedSlice.vision_transformer/transformer_block/add_1:z:01vision_transformer/strided_slice_2/stack:output:03vision_transformer/strided_slice_2/stack_1:output:03vision_transformer/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_maskЮ
Tvision_transformer/sequential_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Щ
Bvision_transformer/sequential_1/layer_normalization_2/moments/meanMean+vision_transformer/strided_slice_2:output:0]vision_transformer/sequential_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(┘
Jvision_transformer/sequential_1/layer_normalization_2/moments/StopGradientStopGradientKvision_transformer/sequential_1/layer_normalization_2/moments/mean:output:0*
T0*'
_output_shapes
:         Ш
Ovision_transformer/sequential_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifference+vision_transformer/strided_slice_2:output:0Svision_transformer/sequential_1/layer_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @в
Xvision_transformer/sequential_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:╔
Fvision_transformer/sequential_1/layer_normalization_2/moments/varianceMeanSvision_transformer/sequential_1/layer_normalization_2/moments/SquaredDifference:z:0avision_transformer/sequential_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(К
Evision_transformer/sequential_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж5Я
Cvision_transformer/sequential_1/layer_normalization_2/batchnorm/addAddV2Ovision_transformer/sequential_1/layer_normalization_2/moments/variance:output:0Nvision_transformer/sequential_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         ╔
Evision_transformer/sequential_1/layer_normalization_2/batchnorm/RsqrtRsqrtGvision_transformer/sequential_1/layer_normalization_2/batchnorm/add:z:0*
T0*'
_output_shapes
:         ъ
Rvision_transformer/sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp[vision_transformer_sequential_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0г
Cvision_transformer/sequential_1/layer_normalization_2/batchnorm/mulMulIvision_transformer/sequential_1/layer_normalization_2/batchnorm/Rsqrt:y:0Zvision_transformer/sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ї
Evision_transformer/sequential_1/layer_normalization_2/batchnorm/mul_1Mul+vision_transformer/strided_slice_2:output:0Gvision_transformer/sequential_1/layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @Ф
Evision_transformer/sequential_1/layer_normalization_2/batchnorm/mul_2MulKvision_transformer/sequential_1/layer_normalization_2/moments/mean:output:0Gvision_transformer/sequential_1/layer_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @т
Nvision_transformer/sequential_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpWvision_transformer_sequential_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0Я
Cvision_transformer/sequential_1/layer_normalization_2/batchnorm/subSubVvision_transformer/sequential_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0Ivision_transformer/sequential_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*'
_output_shapes
:         @Ф
Evision_transformer/sequential_1/layer_normalization_2/batchnorm/add_1AddV2Ivision_transformer/sequential_1/layer_normalization_2/batchnorm/mul_1:z:0Gvision_transformer/sequential_1/layer_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @─
=vision_transformer/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOpFvision_transformer_sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0№
.vision_transformer/sequential_1/dense_1/MatMulMatMulIvision_transformer/sequential_1/layer_normalization_2/batchnorm/add_1:z:0Evision_transformer/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ┬
>vision_transformer/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpGvision_transformer_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ю
/vision_transformer/sequential_1/dense_1/BiasAddBiasAdd8vision_transformer/sequential_1/dense_1/MatMul:product:0Fvision_transformer/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          w
2vision_transformer/sequential_1/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?р
0vision_transformer/sequential_1/dense_1/Gelu/mulMul;vision_transformer/sequential_1/dense_1/Gelu/mul/x:output:08vision_transformer/sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          x
3vision_transformer/sequential_1/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *є╡?щ
4vision_transformer/sequential_1/dense_1/Gelu/truedivRealDiv8vision_transformer/sequential_1/dense_1/BiasAdd:output:0<vision_transformer/sequential_1/dense_1/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:          г
0vision_transformer/sequential_1/dense_1/Gelu/ErfErf8vision_transformer/sequential_1/dense_1/Gelu/truediv:z:0*
T0*'
_output_shapes
:          w
2vision_transformer/sequential_1/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?▐
0vision_transformer/sequential_1/dense_1/Gelu/addAddV2;vision_transformer/sequential_1/dense_1/Gelu/add/x:output:04vision_transformer/sequential_1/dense_1/Gelu/Erf:y:0*
T0*'
_output_shapes
:          ╫
2vision_transformer/sequential_1/dense_1/Gelu/mul_1Mul4vision_transformer/sequential_1/dense_1/Gelu/mul:z:04vision_transformer/sequential_1/dense_1/Gelu/add:z:0*
T0*'
_output_shapes
:          и
2vision_transformer/sequential_1/dropout_4/IdentityIdentity6vision_transformer/sequential_1/dense_1/Gelu/mul_1:z:0*
T0*'
_output_shapes
:          ─
=vision_transformer/sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOpFvision_transformer_sequential_1_dense_2_matmul_readvariableop_resource*
_output_shapes

: 
*
dtype0ю
.vision_transformer/sequential_1/dense_2/MatMulMatMul;vision_transformer/sequential_1/dropout_4/Identity:output:0Evision_transformer/sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
┬
>vision_transformer/sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpGvision_transformer_sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ю
/vision_transformer/sequential_1/dense_2/BiasAddBiasAdd8vision_transformer/sequential_1/dense_2/MatMul:product:0Fvision_transformer/sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
З
IdentityIdentity8vision_transformer/sequential_1/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
╘
NoOpNoOp.^vision_transformer/BroadcastTo/ReadVariableOp&^vision_transformer/add/ReadVariableOp0^vision_transformer/dense/BiasAdd/ReadVariableOp2^vision_transformer/dense/Tensordot/ReadVariableOp?^vision_transformer/sequential_1/dense_1/BiasAdd/ReadVariableOp>^vision_transformer/sequential_1/dense_1/MatMul/ReadVariableOp?^vision_transformer/sequential_1/dense_2/BiasAdd/ReadVariableOp>^vision_transformer/sequential_1/dense_2/MatMul/ReadVariableOpO^vision_transformer/sequential_1/layer_normalization_2/batchnorm/ReadVariableOpS^vision_transformer/sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOpR^vision_transformer/transformer_block/layer_normalization/batchnorm/ReadVariableOpV^vision_transformer/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpT^vision_transformer/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpX^vision_transformer/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpZ^vision_transformer/transformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOp\^vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOpZ^vision_transformer/transformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOp\^vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOp\^vision_transformer/transformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOp^^vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOp\^vision_transformer/transformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOp^^vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOpW^vision_transformer/transformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpY^vision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpQ^vision_transformer/transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOpS^vision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-vision_transformer/BroadcastTo/ReadVariableOp-vision_transformer/BroadcastTo/ReadVariableOp2N
%vision_transformer/add/ReadVariableOp%vision_transformer/add/ReadVariableOp2b
/vision_transformer/dense/BiasAdd/ReadVariableOp/vision_transformer/dense/BiasAdd/ReadVariableOp2f
1vision_transformer/dense/Tensordot/ReadVariableOp1vision_transformer/dense/Tensordot/ReadVariableOp2А
>vision_transformer/sequential_1/dense_1/BiasAdd/ReadVariableOp>vision_transformer/sequential_1/dense_1/BiasAdd/ReadVariableOp2~
=vision_transformer/sequential_1/dense_1/MatMul/ReadVariableOp=vision_transformer/sequential_1/dense_1/MatMul/ReadVariableOp2А
>vision_transformer/sequential_1/dense_2/BiasAdd/ReadVariableOp>vision_transformer/sequential_1/dense_2/BiasAdd/ReadVariableOp2~
=vision_transformer/sequential_1/dense_2/MatMul/ReadVariableOp=vision_transformer/sequential_1/dense_2/MatMul/ReadVariableOp2а
Nvision_transformer/sequential_1/layer_normalization_2/batchnorm/ReadVariableOpNvision_transformer/sequential_1/layer_normalization_2/batchnorm/ReadVariableOp2и
Rvision_transformer/sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOpRvision_transformer/sequential_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2ж
Qvision_transformer/transformer_block/layer_normalization/batchnorm/ReadVariableOpQvision_transformer/transformer_block/layer_normalization/batchnorm/ReadVariableOp2о
Uvision_transformer/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOpUvision_transformer/transformer_block/layer_normalization/batchnorm/mul/ReadVariableOp2к
Svision_transformer/transformer_block/layer_normalization_1/batchnorm/ReadVariableOpSvision_transformer/transformer_block/layer_normalization_1/batchnorm/ReadVariableOp2▓
Wvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOpWvision_transformer/transformer_block/layer_normalization_1/batchnorm/mul/ReadVariableOp2╢
Yvision_transformer/transformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOpYvision_transformer/transformer_block/multi_head_self_attention/key/BiasAdd/ReadVariableOp2║
[vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOp[vision_transformer/transformer_block/multi_head_self_attention/key/Tensordot/ReadVariableOp2╢
Yvision_transformer/transformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOpYvision_transformer/transformer_block/multi_head_self_attention/out/BiasAdd/ReadVariableOp2║
[vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOp[vision_transformer/transformer_block/multi_head_self_attention/out/Tensordot/ReadVariableOp2║
[vision_transformer/transformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOp[vision_transformer/transformer_block/multi_head_self_attention/query/BiasAdd/ReadVariableOp2╛
]vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOp]vision_transformer/transformer_block/multi_head_self_attention/query/Tensordot/ReadVariableOp2║
[vision_transformer/transformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOp[vision_transformer/transformer_block/multi_head_self_attention/value/BiasAdd/ReadVariableOp2╛
]vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOp]vision_transformer/transformer_block/multi_head_self_attention/value/Tensordot/ReadVariableOp2░
Vvision_transformer/transformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOpVvision_transformer/transformer_block/sequential/dense_embed_dim/BiasAdd/ReadVariableOp2┤
Xvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOpXvision_transformer/transformer_block/sequential/dense_embed_dim/Tensordot/ReadVariableOp2д
Pvision_transformer/transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOpPvision_transformer/transformer_block/sequential/dense_mlp/BiasAdd/ReadVariableOp2и
Rvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOpRvision_transformer/transformer_block/sequential/dense_mlp/Tensordot/ReadVariableOp:X T
/
_output_shapes
:           
!
_user_specified_name	input_1
У=
Ц

M__inference_vision_transformer_layer_call_and_return_conditional_losses_36117
input_1
dense_36050:@
dense_36052:@9
#broadcastto_readvariableop_resource:@1
add_readvariableop_resource:A@%
transformer_block_36066:@%
transformer_block_36068:@)
transformer_block_36070:@@%
transformer_block_36072:@)
transformer_block_36074:@@%
transformer_block_36076:@)
transformer_block_36078:@@%
transformer_block_36080:@)
transformer_block_36082:@@%
transformer_block_36084:@%
transformer_block_36086:@%
transformer_block_36088:@)
transformer_block_36090:@ %
transformer_block_36092: )
transformer_block_36094: @%
transformer_block_36096:@ 
sequential_1_36103:@ 
sequential_1_36105:@$
sequential_1_36107:@  
sequential_1_36109: $
sequential_1_36111: 
 
sequential_1_36113:

identityИвBroadcastTo/ReadVariableOpвadd/ReadVariableOpвdense/StatefulPartitionedCallв$sequential_1/StatefulPartitionedCallв)transformer_block/StatefulPartitionedCall<
ShapeShapeinput_1*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┼
rescaling/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_rescaling_layer_call_and_return_conditional_losses_34897Y
Shape_1Shape"rescaling/PartitionedCall:output:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╪
ExtractImagePatchesExtractImagePatches"rescaling/PartitionedCall:output:0*
T0*/
_output_shapes
:         *
ksizes
*
paddingVALID*
rates
*
strides
Z
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :С
Reshape/shapePackstrided_slice_1:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:И
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*4
_output_shapes"
 :                  √
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0dense_36050dense_36052*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_34939В
BroadcastTo/ReadVariableOpReadVariableOp#broadcastto_readvariableop_resource*"
_output_shapes
:@*
dtype0U
BroadcastTo/shape/1Const*
_output_shapes
: *
dtype0*
value	B :U
BroadcastTo/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@Ы
BroadcastTo/shapePackstrided_slice:output:0BroadcastTo/shape/1:output:0BroadcastTo/shape/2:output:0*
N*
T0*
_output_shapes
:Р
BroadcastToBroadcastTo"BroadcastTo/ReadVariableOp:value:0BroadcastTo/shape:output:0*
T0*+
_output_shapes
:         @M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :о
concatConcatV2BroadcastTo:output:0&dense/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  @r
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*"
_output_shapes
:A@*
dtype0o
addAddV2concat:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         A@У
)transformer_block/StatefulPartitionedCallStatefulPartitionedCalladd:z:0transformer_block_36066transformer_block_36068transformer_block_36070transformer_block_36072transformer_block_36074transformer_block_36076transformer_block_36078transformer_block_36080transformer_block_36082transformer_block_36084transformer_block_36086transformer_block_36088transformer_block_36090transformer_block_36092transformer_block_36094transformer_block_36096*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_35637f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ░
strided_slice_2StridedSlice2transformer_block/StatefulPartitionedCall:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*

begin_mask*
end_mask*
shrink_axis_maskъ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0sequential_1_36103sequential_1_36105sequential_1_36107sequential_1_36109sequential_1_36111sequential_1_36113*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_34805|
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
ы
NoOpNoOp^BroadcastTo/ReadVariableOp^add/ReadVariableOp^dense/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:           : : : : : : : : : : : : : : : : : : : : : : : : : : 28
BroadcastTo/ReadVariableOpBroadcastTo/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_1
Т

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_34482

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         A@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Р
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         A@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>к
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         A@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         A@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         A@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         A@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A@:S O
+
_output_shapes
:         A@
 
_user_specified_nameinputs
о	
Ц
,__inference_sequential_1_layer_call_fn_34837
layer_normalization_2_input
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 

	unknown_4:

identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCalllayer_normalization_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_34805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
'
_output_shapes
:         @
5
_user_specified_namelayer_normalization_2_input
■
`
'__inference_dropout_layer_call_fn_38149

inputs
identityИвStatefulPartitionedCall─
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         A * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_34515s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         A `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         A 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         A 
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
C
input_18
serving_default_input_1:0           <
output_10
StatefulPartitionedCall:0         
tensorflow/serving/predict:╕╠
й
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
rescale
	pos_emb

	class_emb

patch_proj

enc_layers
mlp_head

signatures"
_tf_keras_model
ц
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
	24

25"
trackable_list_wrapper
ц
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18
"19
#20
$21
%22
&23
	24

25"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Е
,trace_0
-trace_1
.trace_2
/trace_32Ъ
2__inference_vision_transformer_layer_call_fn_35316
2__inference_vision_transformer_layer_call_fn_36233
2__inference_vision_transformer_layer_call_fn_36290
2__inference_vision_transformer_layer_call_fn_35945╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z,trace_0z-trace_1z.trace_2z/trace_3
ё
0trace_0
1trace_1
2trace_2
3trace_32Ж
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36642
M__inference_vision_transformer_layer_call_and_return_conditional_losses_37029
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36031
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36117╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z0trace_0z1trace_1z2trace_2z3trace_3
╦B╚
 __inference__wrapped_model_34349input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
е
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
:A@2pos_emb
:@2	class_emb
╗
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
'
@0"
trackable_list_wrapper
м
Alayer_with_weights-0
Alayer-0
Blayer_with_weights-1
Blayer-1
Clayer-2
Dlayer_with_weights-2
Dlayer-3
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_sequential
,
Kserving_default"
signature_map
:@2dense/kernel
:@2
dense/bias
J:H@@28transformer_block/multi_head_self_attention/query/kernel
D:B@26transformer_block/multi_head_self_attention/query/bias
H:F@@26transformer_block/multi_head_self_attention/key/kernel
B:@@24transformer_block/multi_head_self_attention/key/bias
J:H@@28transformer_block/multi_head_self_attention/value/kernel
D:B@26transformer_block/multi_head_self_attention/value/bias
H:F@@26transformer_block/multi_head_self_attention/out/kernel
B:@@24transformer_block/multi_head_self_attention/out/bias
": @ 2dense_mlp/kernel
: 2dense_mlp/bias
(:& @2dense_embed_dim/kernel
": @2dense_embed_dim/bias
9:7@2+transformer_block/layer_normalization/gamma
8:6@2*transformer_block/layer_normalization/beta
;:9@2-transformer_block/layer_normalization_1/gamma
::8@2,transformer_block/layer_normalization_1/beta
):'@2layer_normalization_2/gamma
(:&@2layer_normalization_2/beta
 :@ 2dense_1/kernel
: 2dense_1/bias
 : 
2dense_2/kernel
:
2dense_2/bias
 "
trackable_list_wrapper
<
0
1
@2
3"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
МBЙ
2__inference_vision_transformer_layer_call_fn_35316input_1"╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
2__inference_vision_transformer_layer_call_fn_36233x"╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
2__inference_vision_transformer_layer_call_fn_36290x"╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
МBЙ
2__inference_vision_transformer_layer_call_fn_35945input_1"╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
бBЮ
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36642x"╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
бBЮ
M__inference_vision_transformer_layer_call_and_return_conditional_losses_37029x"╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
зBд
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36031input_1"╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
зBд
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36117input_1"╟
╛▓║
FullArgSpec9
args1Ъ.
jself
jx

jtraining
jreturn_attentions
varargs
 
varkw
 
defaultsв
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
э
Strace_02╨
)__inference_rescaling_layer_call_fn_37034в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zStrace_0
И
Ttrace_02ы
D__inference_rescaling_layer_call_and_return_conditional_losses_37042в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zTtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
щ
Ztrace_02╠
%__inference_dense_layer_call_fn_37051в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zZtrace_0
Д
[trace_02ч
@__inference_dense_layer_call_and_return_conditional_losses_37081в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z[trace_0
є
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
batt
cmlp
d
layernorm1
e
layernorm2
fdropout1
gdropout2"
_tf_keras_layer
─
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses
naxis
	!gamma
"beta"
_tf_keras_layer
╗
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
╝
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_random_generator"
_tf_keras_layer
╜
|	variables
}trainable_variables
~regularization_losses
	keras_api
А__call__
+Б&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
J
!0
"1
#2
$3
%4
&5"
trackable_list_wrapper
J
!0
"1
#2
$3
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
э
Зtrace_0
Иtrace_1
Йtrace_2
Кtrace_32·
,__inference_sequential_1_layer_call_fn_34713
,__inference_sequential_1_layer_call_fn_37098
,__inference_sequential_1_layer_call_fn_37115
,__inference_sequential_1_layer_call_fn_34837┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЗtrace_0zИtrace_1zЙtrace_2zКtrace_3
┘
Лtrace_0
Мtrace_1
Нtrace_2
Оtrace_32ц
G__inference_sequential_1_layer_call_and_return_conditional_losses_37158
G__inference_sequential_1_layer_call_and_return_conditional_losses_37208
G__inference_sequential_1_layer_call_and_return_conditional_losses_34857
G__inference_sequential_1_layer_call_and_return_conditional_losses_34877┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЛtrace_0zМtrace_1zНtrace_2zОtrace_3
╩B╟
#__inference_signature_wrapper_36176input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
П	variables
Р	keras_api

Сtotal

Тcount"
_tf_keras_metric
c
У	variables
Ф	keras_api

Хtotal

Цcount
Ч
_fn_kwargs"
_tf_keras_metric
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
▌B┌
)__inference_rescaling_layer_call_fn_37034inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_rescaling_layer_call_and_return_conditional_losses_37042inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
┘B╓
%__inference_dense_layer_call_fn_37051inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЇBё
@__inference_dense_layer_call_and_return_conditional_losses_37081inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ц
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15"
trackable_list_wrapper
Ц
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
ы
Эtrace_0
Юtrace_12░
1__inference_transformer_block_layer_call_fn_37245
1__inference_transformer_block_layer_call_fn_37282╟
╛▓║
FullArgSpec=
args5Ъ2
jself
jinputs

jtraining
jreturn_attention
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЭtrace_0zЮtrace_1
б
Яtrace_0
аtrace_12ц
L__inference_transformer_block_layer_call_and_return_conditional_losses_37535
L__inference_transformer_block_layer_call_and_return_conditional_losses_37816╟
╛▓║
FullArgSpec=
args5Ъ2
jself
jinputs

jtraining
jreturn_attention
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЯtrace_0zаtrace_1
є
б	variables
вtrainable_variables
гregularization_losses
д	keras_api
е__call__
+ж&call_and_return_all_conditional_losses
зquery_dense
и	key_dense
йvalue_dense
кcombine_heads"
_tf_keras_layer
Ю
лlayer_with_weights-0
лlayer-0
мlayer-1
нlayer_with_weights-1
нlayer-2
оlayer-3
п	variables
░trainable_variables
▒regularization_losses
▓	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"
_tf_keras_sequential
╦
╡	variables
╢trainable_variables
╖regularization_losses
╕	keras_api
╣__call__
+║&call_and_return_all_conditional_losses
	╗axis
	gamma
beta"
_tf_keras_layer
╦
╝	variables
╜trainable_variables
╛regularization_losses
┐	keras_api
└__call__
+┴&call_and_return_all_conditional_losses
	┬axis
	gamma
 beta"
_tf_keras_layer
├
├	variables
─trainable_variables
┼regularization_losses
╞	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses
╔_random_generator"
_tf_keras_layer
├
╩	variables
╦trainable_variables
╠regularization_losses
═	keras_api
╬__call__
+╧&call_and_return_all_conditional_losses
╨_random_generator"
_tf_keras_layer
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╤non_trainable_variables
╥layers
╙metrics
 ╘layer_regularization_losses
╒layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
√
╓trace_02▄
5__inference_layer_normalization_2_layer_call_fn_37825в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╓trace_0
Ц
╫trace_02ў
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_37847в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╫trace_0
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╪non_trainable_variables
┘layers
┌metrics
 █layer_regularization_losses
▄layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
э
▌trace_02╬
'__inference_dense_1_layer_call_fn_37856в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▌trace_0
И
▐trace_02щ
B__inference_dense_1_layer_call_and_return_conditional_losses_37874в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▐trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▀non_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
╟
фtrace_0
хtrace_12М
)__inference_dropout_4_layer_call_fn_37879
)__inference_dropout_4_layer_call_fn_37884│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zфtrace_0zхtrace_1
¤
цtrace_0
чtrace_12┬
D__inference_dropout_4_layer_call_and_return_conditional_losses_37889
D__inference_dropout_4_layer_call_and_return_conditional_losses_37901│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zцtrace_0zчtrace_1
"
_generic_user_object
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
|	variables
}trainable_variables
~regularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
э
эtrace_02╬
'__inference_dense_2_layer_call_fn_37910в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zэtrace_0
И
юtrace_02щ
B__inference_dense_2_layer_call_and_return_conditional_losses_37920в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zюtrace_0
 "
trackable_list_wrapper
<
A0
B1
C2
D3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ТBП
,__inference_sequential_1_layer_call_fn_34713layer_normalization_2_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_sequential_1_layer_call_fn_37098inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_sequential_1_layer_call_fn_37115inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
,__inference_sequential_1_layer_call_fn_34837layer_normalization_2_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_1_layer_call_and_return_conditional_losses_37158inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_1_layer_call_and_return_conditional_losses_37208inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
нBк
G__inference_sequential_1_layer_call_and_return_conditional_losses_34857layer_normalization_2_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
нBк
G__inference_sequential_1_layer_call_and_return_conditional_losses_34877layer_normalization_2_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
С0
Т1"
trackable_list_wrapper
.
П	variables"
_generic_user_object
:  (2total
:  (2count
0
Х0
Ц1"
trackable_list_wrapper
.
У	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
b0
c1
d2
e3
f4
g5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
КBЗ
1__inference_transformer_block_layer_call_fn_37245inputs"╟
╛▓║
FullArgSpec=
args5Ъ2
jself
jinputs

jtraining
jreturn_attention
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
1__inference_transformer_block_layer_call_fn_37282inputs"╟
╛▓║
FullArgSpec=
args5Ъ2
jself
jinputs

jtraining
jreturn_attention
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
L__inference_transformer_block_layer_call_and_return_conditional_losses_37535inputs"╟
╛▓║
FullArgSpec=
args5Ъ2
jself
jinputs

jtraining
jreturn_attention
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
L__inference_transformer_block_layer_call_and_return_conditional_losses_37816inputs"╟
╛▓║
FullArgSpec=
args5Ъ2
jself
jinputs

jtraining
jreturn_attention
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
яnon_trainable_variables
Ёlayers
ёmetrics
 Єlayer_regularization_losses
єlayer_metrics
б	variables
вtrainable_variables
гregularization_losses
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
┴2╛╗
▓▓о
FullArgSpec1
args)Ъ&
jself
jinputs
jreturn_attention
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┴2╛╗
▓▓о
FullArgSpec1
args)Ъ&
jself
jinputs
jreturn_attention
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┴
Ї	variables
їtrainable_variables
Ўregularization_losses
ў	keras_api
°__call__
+∙&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
┴
·	variables
√trainable_variables
№regularization_losses
¤	keras_api
■__call__
+ &call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
┴
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
┴
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
┴
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
├
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses
Ш_random_generator"
_tf_keras_layer
┴
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
├
Я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses
е_random_generator"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
п	variables
░trainable_variables
▒regularization_losses
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
х
лtrace_0
мtrace_1
нtrace_2
оtrace_32Є
*__inference_sequential_layer_call_fn_34462
*__inference_sequential_layer_call_fn_37933
*__inference_sequential_layer_call_fn_37946
*__inference_sequential_layer_call_fn_34583┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zлtrace_0zмtrace_1zнtrace_2zоtrace_3
╤
пtrace_0
░trace_1
▒trace_2
▓trace_32▐
E__inference_sequential_layer_call_and_return_conditional_losses_38012
E__inference_sequential_layer_call_and_return_conditional_losses_38092
E__inference_sequential_layer_call_and_return_conditional_losses_34599
E__inference_sequential_layer_call_and_return_conditional_losses_34615┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zпtrace_0z░trace_1z▒trace_2z▓trace_3
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
╡	variables
╢trainable_variables
╖regularization_losses
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
╝	variables
╜trainable_variables
╛regularization_losses
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╜non_trainable_variables
╛layers
┐metrics
 └layer_regularization_losses
┴layer_metrics
├	variables
─trainable_variables
┼regularization_losses
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
╣2╢│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╣2╢│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
╞layer_metrics
╩	variables
╦trainable_variables
╠regularization_losses
╬__call__
+╧&call_and_return_all_conditional_losses
'╧"call_and_return_conditional_losses"
_generic_user_object
╣2╢│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╣2╢│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
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
щBц
5__inference_layer_normalization_2_layer_call_fn_37825inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_37847inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
█B╪
'__inference_dense_1_layer_call_fn_37856inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_dense_1_layer_call_and_return_conditional_losses_37874inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
юBы
)__inference_dropout_4_layer_call_fn_37879inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
)__inference_dropout_4_layer_call_fn_37884inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЙBЖ
D__inference_dropout_4_layer_call_and_return_conditional_losses_37889inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЙBЖ
D__inference_dropout_4_layer_call_and_return_conditional_losses_37901inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
█B╪
'__inference_dense_2_layer_call_fn_37910inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_dense_2_layer_call_and_return_conditional_losses_37920inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
@
з0
и1
й2
к3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╟non_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
Ї	variables
їtrainable_variables
Ўregularization_losses
°__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
_generic_user_object
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╠non_trainable_variables
═layers
╬metrics
 ╧layer_regularization_losses
╨layer_metrics
·	variables
√trainable_variables
№regularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╤non_trainable_variables
╥layers
╙metrics
 ╘layer_regularization_losses
╒layer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
я
рtrace_02╨
)__inference_dense_mlp_layer_call_fn_38101в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zрtrace_0
К
сtrace_02ы
D__inference_dense_mlp_layer_call_and_return_conditional_losses_38139в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
├
чtrace_0
шtrace_12И
'__inference_dropout_layer_call_fn_38144
'__inference_dropout_layer_call_fn_38149│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0zшtrace_1
∙
щtrace_0
ъtrace_12╛
B__inference_dropout_layer_call_and_return_conditional_losses_38154
B__inference_dropout_layer_call_and_return_conditional_losses_38166│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zщtrace_0zъtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
ї
Ёtrace_02╓
/__inference_dense_embed_dim_layer_call_fn_38175в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0
Р
ёtrace_02ё
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_38205в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zёtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
Я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
╟
ўtrace_0
°trace_12М
)__inference_dropout_1_layer_call_fn_38210
)__inference_dropout_1_layer_call_fn_38215│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0z°trace_1
¤
∙trace_0
·trace_12┬
D__inference_dropout_1_layer_call_and_return_conditional_losses_38220
D__inference_dropout_1_layer_call_and_return_conditional_losses_38232│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z∙trace_0z·trace_1
"
_generic_user_object
 "
trackable_list_wrapper
@
л0
м1
н2
о3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ДBБ
*__inference_sequential_layer_call_fn_34462dense_mlp_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
*__inference_sequential_layer_call_fn_37933inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
*__inference_sequential_layer_call_fn_37946inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
*__inference_sequential_layer_call_fn_34583dense_mlp_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_38012inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЦBУ
E__inference_sequential_layer_call_and_return_conditional_losses_38092inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
E__inference_sequential_layer_call_and_return_conditional_losses_34599dense_mlp_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
E__inference_sequential_layer_call_and_return_conditional_losses_34615dense_mlp_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_dense_mlp_layer_call_fn_38101inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_dense_mlp_layer_call_and_return_conditional_losses_38139inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
ьBщ
'__inference_dropout_layer_call_fn_38144inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
'__inference_dropout_layer_call_fn_38149inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЗBД
B__inference_dropout_layer_call_and_return_conditional_losses_38154inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЗBД
B__inference_dropout_layer_call_and_return_conditional_losses_38166inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
уBр
/__inference_dense_embed_dim_layer_call_fn_38175inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_38205inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
юBы
)__inference_dropout_1_layer_call_fn_38210inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
)__inference_dropout_1_layer_call_fn_38215inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЙBЖ
D__inference_dropout_1_layer_call_and_return_conditional_losses_38220inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЙBЖ
D__inference_dropout_1_layer_call_and_return_conditional_losses_38232inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ░
 __inference__wrapped_model_34349Л
	 !"#$%&8в5
.в+
)К&
input_1           
к "3к0
.
output_1"К
output_1         
в
B__inference_dense_1_layer_call_and_return_conditional_losses_37874\#$/в,
%в"
 К
inputs         @
к "%в"
К
0          
Ъ z
'__inference_dense_1_layer_call_fn_37856O#$/в,
%в"
 К
inputs         @
к "К          в
B__inference_dense_2_layer_call_and_return_conditional_losses_37920\%&/в,
%в"
 К
inputs          
к "%в"
К
0         

Ъ z
'__inference_dense_2_layer_call_fn_37910O%&/в,
%в"
 К
inputs          
к "К         
▓
J__inference_dense_embed_dim_layer_call_and_return_conditional_losses_38205d3в0
)в&
$К!
inputs         A 
к ")в&
К
0         A@
Ъ К
/__inference_dense_embed_dim_layer_call_fn_38175W3в0
)в&
$К!
inputs         A 
к "К         A@║
@__inference_dense_layer_call_and_return_conditional_losses_37081v<в9
2в/
-К*
inputs                  
к "2в/
(К%
0                  @
Ъ Т
%__inference_dense_layer_call_fn_37051i<в9
2в/
-К*
inputs                  
к "%К"                  @м
D__inference_dense_mlp_layer_call_and_return_conditional_losses_38139d3в0
)в&
$К!
inputs         A@
к ")в&
К
0         A 
Ъ Д
)__inference_dense_mlp_layer_call_fn_38101W3в0
)в&
$К!
inputs         A@
к "К         A м
D__inference_dropout_1_layer_call_and_return_conditional_losses_38220d7в4
-в*
$К!
inputs         A@
p 
к ")в&
К
0         A@
Ъ м
D__inference_dropout_1_layer_call_and_return_conditional_losses_38232d7в4
-в*
$К!
inputs         A@
p
к ")в&
К
0         A@
Ъ Д
)__inference_dropout_1_layer_call_fn_38210W7в4
-в*
$К!
inputs         A@
p 
к "К         A@Д
)__inference_dropout_1_layer_call_fn_38215W7в4
-в*
$К!
inputs         A@
p
к "К         A@д
D__inference_dropout_4_layer_call_and_return_conditional_losses_37889\3в0
)в&
 К
inputs          
p 
к "%в"
К
0          
Ъ д
D__inference_dropout_4_layer_call_and_return_conditional_losses_37901\3в0
)в&
 К
inputs          
p
к "%в"
К
0          
Ъ |
)__inference_dropout_4_layer_call_fn_37879O3в0
)в&
 К
inputs          
p 
к "К          |
)__inference_dropout_4_layer_call_fn_37884O3в0
)в&
 К
inputs          
p
к "К          к
B__inference_dropout_layer_call_and_return_conditional_losses_38154d7в4
-в*
$К!
inputs         A 
p 
к ")в&
К
0         A 
Ъ к
B__inference_dropout_layer_call_and_return_conditional_losses_38166d7в4
-в*
$К!
inputs         A 
p
к ")в&
К
0         A 
Ъ В
'__inference_dropout_layer_call_fn_38144W7в4
-в*
$К!
inputs         A 
p 
к "К         A В
'__inference_dropout_layer_call_fn_38149W7в4
-в*
$К!
inputs         A 
p
к "К         A ░
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_37847\!"/в,
%в"
 К
inputs         @
к "%в"
К
0         @
Ъ И
5__inference_layer_normalization_2_layer_call_fn_37825O!"/в,
%в"
 К
inputs         @
к "К         @░
D__inference_rescaling_layer_call_and_return_conditional_losses_37042h7в4
-в*
(К%
inputs           
к "-в*
#К 
0           
Ъ И
)__inference_rescaling_layer_call_fn_37034[7в4
-в*
(К%
inputs           
к " К           ╚
G__inference_sequential_1_layer_call_and_return_conditional_losses_34857}!"#$%&LвI
Bв?
5К2
layer_normalization_2_input         @
p 

 
к "%в"
К
0         

Ъ ╚
G__inference_sequential_1_layer_call_and_return_conditional_losses_34877}!"#$%&LвI
Bв?
5К2
layer_normalization_2_input         @
p

 
к "%в"
К
0         

Ъ │
G__inference_sequential_1_layer_call_and_return_conditional_losses_37158h!"#$%&7в4
-в*
 К
inputs         @
p 

 
к "%в"
К
0         

Ъ │
G__inference_sequential_1_layer_call_and_return_conditional_losses_37208h!"#$%&7в4
-в*
 К
inputs         @
p

 
к "%в"
К
0         

Ъ а
,__inference_sequential_1_layer_call_fn_34713p!"#$%&LвI
Bв?
5К2
layer_normalization_2_input         @
p 

 
к "К         
а
,__inference_sequential_1_layer_call_fn_34837p!"#$%&LвI
Bв?
5К2
layer_normalization_2_input         @
p

 
к "К         
Л
,__inference_sequential_1_layer_call_fn_37098[!"#$%&7в4
-в*
 К
inputs         @
p 

 
к "К         
Л
,__inference_sequential_1_layer_call_fn_37115[!"#$%&7в4
-в*
 К
inputs         @
p

 
к "К         
└
E__inference_sequential_layer_call_and_return_conditional_losses_34599wDвA
:в7
-К*
dense_mlp_input         A@
p 

 
к ")в&
К
0         A@
Ъ └
E__inference_sequential_layer_call_and_return_conditional_losses_34615wDвA
:в7
-К*
dense_mlp_input         A@
p

 
к ")в&
К
0         A@
Ъ ╖
E__inference_sequential_layer_call_and_return_conditional_losses_38012n;в8
1в.
$К!
inputs         A@
p 

 
к ")в&
К
0         A@
Ъ ╖
E__inference_sequential_layer_call_and_return_conditional_losses_38092n;в8
1в.
$К!
inputs         A@
p

 
к ")в&
К
0         A@
Ъ Ш
*__inference_sequential_layer_call_fn_34462jDвA
:в7
-К*
dense_mlp_input         A@
p 

 
к "К         A@Ш
*__inference_sequential_layer_call_fn_34583jDвA
:в7
-К*
dense_mlp_input         A@
p

 
к "К         A@П
*__inference_sequential_layer_call_fn_37933a;в8
1в.
$К!
inputs         A@
p 

 
к "К         A@П
*__inference_sequential_layer_call_fn_37946a;в8
1в.
$К!
inputs         A@
p

 
к "К         A@╛
#__inference_signature_wrapper_36176Ц
	 !"#$%&Cв@
в 
9к6
4
input_1)К&
input_1           "3к0
.
output_1"К
output_1         
╩
L__inference_transformer_block_layer_call_and_return_conditional_losses_37535z ;в8
1в.
$К!
inputs         A@
p 
p 
к ")в&
К
0         A@
Ъ ╩
L__inference_transformer_block_layer_call_and_return_conditional_losses_37816z ;в8
1в.
$К!
inputs         A@
p
p 
к ")в&
К
0         A@
Ъ в
1__inference_transformer_block_layer_call_fn_37245m ;в8
1в.
$К!
inputs         A@
p 
p 
к "К         A@в
1__inference_transformer_block_layer_call_fn_37282m ;в8
1в.
$К!
inputs         A@
p
p 
к "К         A@╫
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36031Е
	 !"#$%&@в=
6в3
)К&
input_1           
p 
p 
к "%в"
К
0         

Ъ ╫
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36117Е
	 !"#$%&@в=
6в3
)К&
input_1           
p
p 
к "%в"
К
0         

Ъ ╨
M__inference_vision_transformer_layer_call_and_return_conditional_losses_36642
	 !"#$%&:в7
0в-
#К 
x           
p 
p 
к "%в"
К
0         

Ъ ╨
M__inference_vision_transformer_layer_call_and_return_conditional_losses_37029
	 !"#$%&:в7
0в-
#К 
x           
p
p 
к "%в"
К
0         

Ъ о
2__inference_vision_transformer_layer_call_fn_35316x
	 !"#$%&@в=
6в3
)К&
input_1           
p 
p 
к "К         
о
2__inference_vision_transformer_layer_call_fn_35945x
	 !"#$%&@в=
6в3
)К&
input_1           
p
p 
к "К         
и
2__inference_vision_transformer_layer_call_fn_36233r
	 !"#$%&:в7
0в-
#К 
x           
p 
p 
к "К         
и
2__inference_vision_transformer_layer_call_fn_36290r
	 !"#$%&:в7
0в-
#К 
x           
p
p 
к "К         
