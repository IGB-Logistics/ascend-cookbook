FROM ubuntu:22.04

# apt-get update
RUN sed -i s@/ports.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -y && apt-get install net-tools vim -y

# bashrc
RUN echo "PATH=\"/root/miniconda3/bin\":$PATH" >> .bashrc \
    && echo "DRV_LIB64_COMMON_LDPATH=\"/usr/local/Ascend/driver/lib64/common\"" >> /root/.bashrc \
    && echo "DRV_LIB64_DRV_LDPATH=\"/usr/local/Ascend/driver/lib64/driver\"" >> /root/.bashrc \
    && echo "DRV_LIB64_LDPATH=\"/usr/local/Ascend/driver/lib64\"" >> /root/.bashrc \
    && echo "export LD_LIBRARY_PATH=\"${DRV_LIB64_COMMON_LDPATH}\":\"${DRV_LIB64_DRV_LDPATH}\":\"${DRV_LIB64_LDPATH}\":\"${LD_LIBRARY_PATH}\"" >> /root/.bashrc \
    && echo "export ASCEND_TENSOR_COMPILER_INCLUDE=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include" >> /root/.bashrc \
    && echo "export PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/:$PATH" >> /root/.bashrc

# miniconda
WORKDIR /app
COPY Miniconda3-latest-Linux-aarch64.sh /app/
RUN chmod +x Miniconda3-latest-Linux-aarch64.sh

RUN ./Miniconda3-latest-Linux-aarch64.sh -b
ENV PATH /root/miniconda3/bin:$PATH
RUN conda create -n llm python=3.9 -y && conda init 
RUN exec bash && conda activate llm 
ENV PATH /root/miniconda3/envs/llm/bin/:$PATH
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN python -V

# cann
COPY Ascend-cann-toolkit_7.0.0_linux-aarch64.run Ascend-cann-nnae_7.0.0_linux-aarch64.run Ascend-cann-kernels-910b_7.0.0_linux.run /app/
RUN chmod +x Ascend-cann-toolkit_7.0.0_linux-aarch64.run Ascend-cann-nnae_7.0.0_linux-aarch64.run Ascend-cann-kernels-910b_7.0.0_linux.run
RUN echo "y" | ./Ascend-cann-toolkit_7.0.0_linux-aarch64.run --install
RUN echo "y" | ./Ascend-cann-nnae_7.0.0_linux-aarch64.run --install
ENV ASCEND_HOME_PATH /usr/local/Ascend/ascend-toolkit/latest
RUN echo "y" | ./Ascend-cann-kernels-910b_7.0.0_linux.run --install
RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc


# pip install
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN echo "export LD_PRELOAD=\"/root/miniconda3/envs/llm/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0 /root/miniconda3/envs/llm/lib/libgomp.so.1\""  >> /root/.bashrc
