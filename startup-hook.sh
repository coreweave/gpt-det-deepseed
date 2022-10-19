export PYTHONPATH=$PYTHONPATH:/gpt-ds

mkdir -p /gpt-ds

cp /run/determined/workdir/gpt_neox_config/small_bf16.yml /gpt-ds
cp -ar /run/determined/workdir/gpt-neox/* /gpt-ds
# cp -a /run/determined/workdir/launch.sh /gpt-ds
