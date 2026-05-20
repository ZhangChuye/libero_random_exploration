#!/bin/bash
source ~/.bashrc
# source activate v2a
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/coc/flash7/czhang883/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/coc/flash7/czhang883/miniconda3/etc/profile.d/conda.sh"
fi
conda activate v2a

export PYTHONPATH=/coc/flash7/czhang883/Documents/LIBERO:$PYTHONPATH


# sub_conf='lb_randsam_8tk_perTk5'
# sub_conf='lb_randsam_8tk_perTk500'
# sub_conf='lb_randsam_goal_20ep'
# sub_conf='lb_randsam_goal_40ep'
# sub_conf='lb_randsam_goal_80ep'
# sub_conf='lb_randsam_goal_3ep'
# sub_conf='lb_randsam_goal_500ep'
sub_conf='lb_randsam_spatial_3ep'
# sub_conf='lb_randsam_spatial_50ep'
# sub_conf='lb_randsam_goal_example'

{
MUJOCO_EGL_DEVICE_ID=${1:-0} \
MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=${1:-0} \
python environment/libero/lb_data/lb_randsam.py \
    --sub_conf ${sub_conf}

exit 0
}

