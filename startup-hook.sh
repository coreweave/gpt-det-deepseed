export PYTHONPATH=$PYTHONPATH:/gpt-ds

mkdir -p /gpt-ds

cp /run/determined/workdir/gpt_neox_config/small.yml /gpt-ds
cp /run/determined/workdir/train_deepspeed_launcher.py /gpt-ds
cp -ar /run/determined/workdir/gpt-neox/* /gpt-ds
cd /gpt-ds
