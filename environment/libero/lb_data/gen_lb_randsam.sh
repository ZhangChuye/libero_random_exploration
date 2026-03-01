#!/bin/bash
source ~/.bashrc
# source activate v2a
source ~/miniconda3/etc/profile.d/conda.sh
conda activate v2a

export PYTHONPATH=/media/czhang883/PORTABLE_SSD/GVF-TAPE/third_party/LIBERO:$PYTHONPATH


# sub_conf='lb_randsam_8tk_perTk5'
# sub_conf='lb_randsam_8tk_perTk500'
# sub_conf='lb_randsam_goal_20ep'
# sub_conf='lb_randsam_goal_40ep'
# sub_conf='lb_randsam_goal_80ep'
sub_conf='lb_randsam_goal_3ep'
# sub_conf='lb_randsam_goal_example'

{
MUJOCO_EGL_DEVICE_ID=${1:-0} \
MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=${1:-0} \
python environment/libero/lb_data/lb_randsam.py \
    --sub_conf ${sub_conf}

exit 0
}

