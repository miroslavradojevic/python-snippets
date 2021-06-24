import ruamel.yaml
import argparse
from os.path import abspath, join, dirname, splitext, exists

parser = argparse.ArgumentParser(description='(anaconda) environment.yml -> (pip) requirements.txt')
parser.add_argument('environment_yml_file', type=str, metavar='environment.yml', default='environment.yml', help='the input file')
parser.add_argument('-o', type=str, metavar='requirements.txt', default=None, help='the output file')
args = parser.parse_args()

if not exists(args.environment_yml_file):
    print("Environment file not found")
    exit()

out_file = args.o

if out_file == None:
    yml_parent = dirname(args.environment_yml_file)
    yml_fname = splitext(args.environment_yml_file)[0]
    out_file = abspath(join(yml_parent, yml_fname + "_requirements.txt"))

print(args.environment_yml_file, '->', out_file)

yaml = ruamel.yaml.YAML()
data = yaml.load(open(args.environment_yml_file))

requirements = []
for dep in data['dependencies']:
    if isinstance(dep, str):
        arr = dep.split('=')
        package = arr[0]
        package_version = arr[1]
        if len(arr) == 3:
            python_version = arr[2]
            if python_version == '0':
                continue
        if package != 'python':
            requirements.append(package + '==' + package_version)
    elif isinstance(dep, dict):
        for preq in dep.get('pip', []):
            requirements.append(preq)

with open(out_file, 'w') as fp:
    for requirement in requirements:
       print(requirement, file=fp)

print('Intall dependencies within the right python environment using:')
print('pip install -r '+out_file)