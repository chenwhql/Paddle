#!/bin/bash

# This script is used to grep invalid PADDLE checks by directory or file in the paddle/fluid/,
#   the result show all invalid PADDLE checks in specified directory or file.

# Usage: 
#   - bash grep_invalid_enforce.sh [target directory or file] (run in tools directory)
#       - The default check path is paddle/fluid/operators

# Result Examples:
# 1. grep invalid PADDLE checks in directory

#     - Command: /work/paddle/tools {develop} bash grep_invalid_enforce.sh ../paddle/fluid/imperative 
#     - Results:
#         - paddle/fluid/imperative/gradient_accumulator.cc
#         PADDLE_ENFORCE_EQ(dst_tensor->numel() == numel, true,
#                             "dst_numel %d vs. src_numel %d", dst_tensor->numel(),
#                             numel);

#         - paddle/fluid/imperative/nccl_context.cc
#         PADDLE_ENFORCE_EQ(addr.size(), 2UL,
#                             "The endpoint should contain host and port: %s", ep);
#         PADDLE_THROW("create server fd failed");
#         PADDLE_THROW("set socket opt failed");
#         PADDLE_THROW("binding failed on ep: %s", ep);
#         PADDLE_THROW("listen on server fd failed");
#         PADDLE_THROW("accept the new socket fd failed");
#         PADDLE_THROW("reading the ncclUniqueId from socket failed");
#         PADDLE_ENFORCE_EQ(addr.size(), 2UL,
#                             "The endpoint should contain host and port: %s", ep);
#         PADDLE_THROW("create socket failed");
#         PADDLE_THROW("invalied address: %s", ep);

#         - paddle/fluid/imperative/jit/program_desc_tracer.cc
#         PADDLE_ENFORCE_NOT_NULL(new_var);
#         PADDLE_ENFORCE_EQ(inner_var.IsInitialized(), true);
#         PADDLE_THROW("Not support variable type %s",
#                         framework::ToTypeName(inner_var.Type()));

# 2. grep invalid PADDLE checks in file

#     - Command: /work/paddle/tools {develop} bash grep_invalid_enforce.sh ../paddle/fluid/pybind/reader_py.cc
#     - Results:
#         - paddle/fluid/pybind/reader_py.cc
#         PADDLE_THROW(
#                     "Place cannot be CUDAPlace when use_double_buffer is False");
#         PADDLE_ENFORCE_NOT_NULL(exceptions_[i]);
#         PADDLE_ENFORCE_EQ(status, Status::kException);
#         PADDLE_ENFORCE_EQ(status, Status::kSuccess);    

ROOT_DIR=../paddle/fluid/operators

if [ "$1" != "" ]; then
    ROOT_DIR=$1
fi

white_list_str="\
    layer_norm_op.cc \
    box_clip_op.cc \
    box_clip_op.h \
    random_crop_op.h \
    elementwise_op_function.cu.h \
    fused_elemwise_activation_op.cc \
    auc_op.cu"

function enforce_scan(){
    paddle_check=`grep -zoE "(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\(.[^,\);]*.[^;]*\);\s" $1 || true`
    valid_check=`echo "$paddle_check" | grep -zoE '(PADDLE_ENFORCE[A-Z_]{0,9}|PADDLE_THROW)\((.[^,;]+,)*.[^";]*(errors::).[^"]*".[^";]{20,}.[^;]*\);\s' || true`
    invalid_check=`echo "$paddle_check" | grep -vxF "$valid_check" || true`
    if [ "${invalid_check}" != "" ];then
        file_path=$1
        echo -e "\n- ${file_path#../}"
        echo "${invalid_check}"
    fi
}

function walk_dir(){
    local i=0
    local dir_array
    for file in `ls $1`
    do
        if [ -f $1"/"$file ];then
            in_white_list=$(echo $white_list_str | grep "${file}")
            if [[ "$in_white_list" == "" ]];then
                enforce_scan $1"/"$file
            fi
        fi
        if [ -d $1"/"$file ];then
            dir_array[$i]=$1"/"$file
            ((i++))
        fi
    done
    for sub_dir_name in ${dir_array[@]}
    do
        walk_dir $sub_dir_name
    done
}

function walk_file(){
    file_path=$1
    file_name=`echo ${file_path##*/} `
    if [ -f $file_path ];then
        in_white_list=$(echo $white_list_str | grep "${file_name}")
        if [[ "$in_white_list" == "" ]];then
            enforce_scan $file_path
        fi
    fi
}

if [ -f $ROOT_DIR ];then
    walk_file $ROOT_DIR
else
    walk_dir $ROOT_DIR
fi
