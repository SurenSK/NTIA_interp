FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y tcpdump && rm -rf /var/lib/apt/lists/*

# 1. Install Dependencies & Repositories
RUN apt-get update && apt-get install -y \
    gnupg git curl wget tcpdump ca-certificates \
    iperf3 build-essential cmake ninja-build \
    gcc g++ pkg-config iproute2 iputils-ping \
    libfftw3-dev libmbedtls-dev libsctp-dev \
    libyaml-cpp-dev libgtest-dev libzmq3-dev \
    libboost-program-options-dev libconfig++-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# MongoDB 8.0
RUN curl -fsSL https://pgp.mongodb.com/server-8.0.asc | gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg --dearmor \
    && echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-8.0.list \
    && apt-get update && apt-get install -y mongodb-org \
    && rm -rf /var/lib/apt/lists/*

# Open5GS (Latest)
RUN wget -qO - "https://build.opensuse.org/projects/home:acetcom/signing_keys/download?kind=gpg" | apt-key add - \
    && sh -c "echo 'deb http://download.opensuse.org/repositories/home:/acetcom:/open5gs:/latest/xUbuntu_22.04/ ./' > /etc/apt/sources.list.d/open5gs.list" \
    && apt-get update && apt-get install -y open5gs \
    && rm -rf /var/lib/apt/lists/*

# 2. Compile srsRAN
WORKDIR /usr/src

# srsRAN Project (gNB: CU/DU)
RUN git clone https://github.com/srsRAN/srsRAN_Project.git \
    && cd srsRAN_Project \
    && cmake -S . -B build -GNinja -DCMAKE_INSTALL_PREFIX=/usr -DENABLE_EXPORT=ON -DENABLE_ZEROMQ=ON \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && cd .. && rm -rf srsRAN_Project

# srsRAN 4G (UE)
RUN git clone https://github.com/srsRAN/srsRAN_4G.git \
    && cd srsRAN_4G \
    && cmake -S . -B build -GNinja -DCMAKE_INSTALL_PREFIX=/usr \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && cd .. && rm -rf srsRAN_4G

# 3. Configuration & Provisioning

# Patch Open5GS Configs (MCC/MNC/TAC)
RUN for i in /etc/open5gs/*.yaml; do sed -i -e 's/mcc: 999/mcc: 901/' -e 's/tac: 1/tac: 7/' -e '/cafe::/d' "$i"; done

# Pre-seed Database Script
RUN curl -sL https://github.com/open5gs/open5gs/raw/main/misc/db/open5gs-dbctl -o /usr/bin/open5gs-dbctl \
    && chmod +x /usr/bin/open5gs-dbctl

# Create Subscriber Provisioning Script
RUN cat <<EOF > /usr/bin/add_subscriber.sh
#!/bin/bash
set -e

echo "Waiting for MongoDB to start..."
until mongosh --eval "printjson(db.runCommand('ping'))" --quiet > /dev/null 2>&1; do
    sleep 1
done

echo "MongoDB is up. Adding subscriber..."
/usr/bin/open5gs-dbctl add 901700000000001 00112233445566778899aabbccddeeff 63bfa50ee6523365ff14c1f45f88737d

echo "Subscriber added successfully."
touch /tmp/db_ready
EOF
RUN chmod +x /usr/bin/add_subscriber.sh

# Create Network Namespace Setup Script
RUN cat <<EOF > /usr/bin/setup_netns.sh
#!/bin/bash
set -e

if ! ip netns list | grep -q "ue1"; then
    echo "Creating network namespace ue1..."
    ip netns add ue1
fi

ip netns exec ue1 ip link set lo up
echo "Network namespace ue1 is ready."
EOF
RUN chmod +x /usr/bin/setup_netns.sh

# Create Core Network Setup Script
RUN cat <<EOF > /usr/bin/setup_ogstun.sh
#!/bin/bash
set -e

until ip link show ogstun >/dev/null 2>&1; do sleep 0.5; done

ip addr add 10.45.0.1/16 dev ogstun
ip link set up dev ogstun

ip addr show ogstun
echo "Network interface ogstun is ready."
EOF
RUN chmod +x /usr/bin/setup_ogstun.sh

# Create necessary directories for mounts and logs
RUN mkdir -p /var/log/supervisor /config /output

# Inject Supervisord Config
# Note: Paths updated to use /output for pcaps and /config for srsRAN yamls
RUN cat <<EOF > /etc/supervisor/conf.d/supervisord.conf
[unix_http_server]
file=/var/run/supervisor.sock

[supervisord]
logfile=/var/log/supervisor/supervisord.log
logfile_maxbytes=50MB
logfile_backups=10
loglevel=info
pidfile=/var/run/supervisord.pid
nodaemon=true
minfds=1024
minprocs=200

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock

[program:pcap_recorder]
command=tcpdump -i any -w /output/wire_trace.pcap -U sctp
autostart=true
autorestart=true
priority=1

[program:netns-init]
command=/usr/bin/setup_netns.sh
priority=5
autostart=true
autorestart=false
startretries=0
stdout_logfile=/var/log/supervisor/netns-init.out.log
stderr_logfile=/var/log/supervisor/netns-init.err.log

[program:tcpdump]
command=tcpdump -i lo port 38472 -w /output/cu_f1c.pcap
priority=5
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/tcpdump.out.log
stderr_logfile=/var/log/supervisor/tcpdump.err.log

[program:mongod]
command=/usr/bin/mongod -f /etc/mongod.conf
environment=MONGODB_CONFIG_OVERRIDE_NOFORK="1"
priority=10
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/mongod.out.log
stderr_logfile=/var/log/supervisor/mongod.err.log

[program:subscriber-init]
command=/usr/bin/add_subscriber.sh
priority=15
autostart=true
autorestart=false
startretries=0
stdout_logfile=/var/log/supervisor/subscriber-init.out.log
stderr_logfile=/var/log/supervisor/subscriber-init.err.log

[program:open5gs-nrf]
command=open5gs-nrfd -c /etc/open5gs/nrf.yaml
priority=20
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/open5gs-nrf.out.log
stderr_logfile=/var/log/supervisor/open5gs-nrf.err.log

[program:open5gs-scp]
command=open5gs-scpd -c /etc/open5gs/scp.yaml
priority=21
autostart=true
autorestart=true

[program:open5gs-sepp]
command=open5gs-seppd -c /etc/open5gs/sepp.yaml
priority=21
autostart=true
autorestart=true

[program:open5gs-ausf]
command=open5gs-ausfd -c /etc/open5gs/ausf.yaml
priority=21
autostart=true
autorestart=true

[program:open5gs-udm]
command=open5gs-udmd -c /etc/open5gs/udm.yaml
priority=21
autostart=true
autorestart=true

[program:open5gs-pcf]
command=open5gs-pcfd -c /etc/open5gs/pcf.yaml
priority=21
autostart=true
autorestart=true

[program:open5gs-nssf]
command=open5gs-nssfd -c /etc/open5gs/nssf.yaml
priority=21
autostart=true
autorestart=true

[program:open5gs-bsf]
command=open5gs-bsfd -c /etc/open5gs/bsf.yaml
priority=21
autostart=true
autorestart=true

[program:open5gs-udr]
command=open5gs-udrd -c /etc/open5gs/udr.yaml
priority=21
autostart=true
autorestart=true

[program:open5gs-mme]
command=open5gs-mmed -c /etc/open5gs/mme.yaml
priority=22
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/open5gs-mme.out.log
stderr_logfile=/var/log/supervisor/open5gs-mme.err.log

[program:open5gs-sgwc]
command=open5gs-sgwcd -c /etc/open5gs/sgwc.yaml
priority=22
autostart=true
autorestart=true

[program:open5gs-smf]
command=open5gs-smfd -c /etc/open5gs/smf.yaml
priority=22
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/open5gs-smf.out.log
stderr_logfile=/var/log/supervisor/open5gs-smf.err.log

[program:open5gs-amf]
command=open5gs-amfd -c /etc/open5gs/amf.yaml
priority=22
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/open5gs-amf.out.log
stderr_logfile=/var/log/supervisor/open5gs-amf.err.log

[program:open5gs-sgwu]
command=open5gs-sgwud -c /etc/open5gs/sgwu.yaml
priority=23
autostart=true
autorestart=true

[program:open5gs-upf]
command=open5gs-upfd -c /etc/open5gs/upf.yaml
priority=23
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/open5gs-upf.out.log
stderr_logfile=/var/log/supervisor/open5gs-upf.err.log

[program:open5gs-hss]
command=open5gs-hssd -c /etc/open5gs/hss.yaml
priority=23
autostart=true
autorestart=true

[program:open5gs-pcrf]
command=open5gs-pcrfd -c /etc/open5gs/pcrf.yaml
priority=23
autostart=true
autorestart=true

[group:open5gs]
programs=open5gs-nrf,open5gs-scp,open5gs-sepp,open5gs-ausf,open5gs-udm,open5gs-pcf,open5gs-nssf,open5gs-bsf,open5gs-udr,open5gs-mme,open5gs-sgwc,open5gs-smf,open5gs-amf,open5gs-sgwu,open5gs-upf,open5gs-hss,open5gs-pcrf

[program:ogstun-init]
command=/usr/bin/setup_ogstun.sh
priority=25
autostart=true
autorestart=false
startretries=0
stdout_logfile=/var/log/supervisor/ogstun-init.out.log
stderr_logfile=/var/log/supervisor/ogstun-init.err.log

[program:srscu]
command=srscu -c /config/cu.yml
priority=30
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/srscu.out.log
stderr_logfile=/var/log/supervisor/srscu.err.log

[program:srsdu]
command=srsdu -c /config/du.yml
priority=31
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/srsdu.out.log
stderr_logfile=/var/log/supervisor/srsdu.err.log

[program:srsue]
command=bash -c "while [ ! -f /tmp/db_ready ]; do sleep 0.1; done; exec srsue /config/ue.conf"
priority=32
autostart=true
autorestart=true
stdout_logfile=/output/ue.log
stderr_logfile=/output/ue.err

[group:srsran]
programs=srscu,srsdu,srsue

[program:iperf-server]
command=iperf3 -s
priority=40
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/iperf-server.out.log

[program:iperf-client]
command=bash -c "echo 'Waiting for UE connectivity...'; until ip netns exec ue1 ping -c 1 -W 1 10.45.0.1 > /dev/null 2>&1; do sleep 2; done; echo 'UE Connected! Starting traffic...'; ip netns exec ue1 iperf3 -t 5 -c 10.45.0.1; echo 'Finished. Stopping container...'; kill -SIGTERM 1"
priority=50
autostart=true
autorestart=false
startretries=0
stdout_logfile=/output/iperf-client.log
stderr_logfile=/var/log/supervisor/iperf-client.err.log
EOF

ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
