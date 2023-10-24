import tools
import time

ENV = "gridworld"
MACHINE = "regular"
config_file = {
    "gridworld": "gridworld.json",
}
machine_cmd = {
    "regular": "optirun python3 -B ",
    "vaughan": "/usr/local/bin/srun --gres gpu:1 -c 8 "+\
        "--mem 20G -p p100,t4v1,t4v2 /usr/bin/xvfb-run -a /h/ashishg/.miniconda3/envs/main/bin/python3 -B ",
}[MACHINE]
config = tools.data.Configuration({
    "cmds": [
        machine_cmd + \
        ("[file] -c configs/%s -seed [seed] -beta [beta]" % config_file[ENV]),
    ],
    "vars": {
        "file": [
            "prob_icl.py",
        ],
        "seed": [
            0,
        ],
        "beta": [
            -1. # use the one defined in the config file
        ],
    },
    "logger": tools.data.Logger(),
})
pm = tools.run.ProcessManager(config)
pm.launch()
while True:
    pm.check()
    time.sleep(1)
