FROM coreweave/nccl-tests:2022-10-16_00-04-11.692_EDT

# Run updates and install packages for build
RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends software-properties-common && \
    add-apt-repository universe && \
    apt-get -qq update && \
    apt-get install -y --no-install-recommends \
       python3-pip \
       python3-dev \
       git \
       python3-setuptools \
       python3-setuptools-rust \
       rustc && \
    apt-get clean

# #### Python packages
# RUN pip3 install --upgrade pip
# RUN pip3 install --no-cache-dir torch==1.11.0+cu113 \
#     -f https://download.pytorch.org/whl/torch_stable.html
# COPY requirements/requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt
# COPY requirements/requirements-onebitadam.txt .
# RUN pip3 install --no-cache-dir -r requirements-onebitadam.txt
# COPY requirements/requirements-sparseattention.txt .
# RUN pip3 install --no-cache-dir -r requirements-sparseattention.txt

# ## Install APEX
# RUN pip install --no-cache-dir -v --disable-pip-version-check --no-cache-dir \
#     --global-option="--cpp_ext" --global-option="--cuda_ext" \
#     git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597