FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y \
        protobuf-compiler \
        sudo \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* 



COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && rm /tmp/environment.yml

ARG username
ARG userid

ARG home=/home/${username}
ARG workdir=${home}/project

RUN adduser ${username} --uid ${userid} --gecos '' --disabled-password \
    && echo "${username} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${username} \
    && chmod 0440 /etc/sudoers.d/${username}

WORKDIR ${workdir}
RUN chown ${username}:${username} ${workdir}

USER ${username}
WORKDIR ${workdir}

ENV PATH /opt/conda/envs/project/bin:$PATH

RUN git-nbdiffdriver config --enable --global

RUN git config --global diff.jupyternotebook.command 'git-nbdiffdriver diff --ignore-details'

COPY docker/bashrc.bash /tmp/
RUN cat /tmp/bashrc.bash >> ${home}/.bashrc \
    && echo "export PATH=\"${workdir}/docker/bin:$PATH\"" >> ${home}/.bashrc \
    && sudo rm /tmp/bashrc.bash
