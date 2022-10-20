export PYTHONPATH=$PYTHONPATH:/gpt-ds

mkdir -p /gpt-ds

cp /run/determined/workdir/gpt_neox_config/small.yml /gpt-ds
cp -ar /run/determined/workdir/gpt-neox/* /gpt-ds
