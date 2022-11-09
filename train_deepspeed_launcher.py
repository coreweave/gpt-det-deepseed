# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train"""
from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain
import glob
import shutil
import os
import determined as det

if __name__ == "__main__":
    distributed = det.core.DistributedContext.from_deepspeed()

    with det.core.init(distributed=distributed) as context:
        shared_hostfile = "/mnt/finetune-gpt-neox/hostfile.txt"
        if os.environ["RANK"]=="0":
            try:
                os.remove(shared_hostfile)
            except:
                pass
            assert not os.path.exists(shared_hostfile)
            hostfile = glob.glob("/tmp/hostfile*.txt")[0]
            shutil.copyfile(hostfile, shared_hostfile)

        # We synchronize here to make sure shared_hostfile is ready for all nodes.
        context.distributed.broadcast(1)
        assert os.path.exists(shared_hostfile)

        neox_args = NeoXArgs.from_ymls(
            [
                # "determined_cluster.yml",
                "small.yml",
            ],
            overwrite_values={'hostfile': shared_hostfile}
        )
        neox_args.configure_distributed_args()
        neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab
        neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined
        print(neox_args)
        pretrain(neox_args=neox_args)