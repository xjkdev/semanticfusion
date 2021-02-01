import re
import os
import collections

dirname = os.path.dirname(os.path.abspath(__file__))
file = open(os.path.join(dirname, '../model.py'), 'r')
output_file = open(os.path.join(dirname, '../../src/cnn_interface/model.h'), 'w')

lines = file.readlines()
lines = [''] + lines  # lines count from 1

init_begins = -1
init_ends = -1
forward_begins = -1
forward_ends = -1

for i, line in enumerate(lines):
    if 'super().__init__()' in line:
        init_begins = i + 1
    if 'def forward(self, x):' in line:
        init_ends = i
        forward_begins = i + 1
    if "if __name__ == '__main__':" in line:
        forward_ends = i - 1

print("init function are in lines", init_begins, ':', init_ends)
print("forward function are in lines", forward_begins, ":", forward_ends)

assert forward_ends > forward_begins > init_ends > init_begins >= 0, 'init or forward not found'

# translate init part
init_part = lines[init_begins:init_ends]
init_result = []
var_names = collections.defaultdict(list)
for line in init_part:
    m = re.match(r'\s+self\.(\w+)\s+=\s+nn\.(\w+)\((.*)\)\s+$', line)
    if not m:
        if line.strip() == '':
            init_result.append('\n')
        else:
            init_result.append('    //' + line.strip() + '\n')
        continue

    var, type, param = m[1], m[2], m[3]
    if '=' in param:
        allargs = param.split(',')
        args = list(x for x in allargs if '=' not in x)
        kwargs = list(x.split('=') for x in allargs if '=' in x)
        kparam = ''
        for k, v in kwargs:
            k = k.strip()
            if v == 'True':
                v = 'true'
            elif v == 'False':
                v = 'false'
            if k == 'kernel_size' or 'channels' in k:
                args.append(v)
            elif k == 'return_indices':
                continue
            else:
                kparam += f'.{k}({v})'
        param = f'torch::nn::{type}Options({", ".join(args)})' + kparam
    init_result.append(
        f'    {var} = register_module("{var}", torch::nn::{type}({param}));\n')
    if var not in var_names[f'torch::nn::{type}']:
        var_names[f'torch::nn::{type}'].append(var)

# translate forward part
forward_part = lines[forward_begins:forward_ends]
forward_result = []
for line in forward_part:
    if 'return' in line:
        forward_result.append(line.strip() + '; \n')
    elif '=' in line:
        line = line.strip()
        m1 = re.match(
            r'(down\d),\s+(mask\d)\s+=\s+self\.(pool\d)\((down\w+)\)',
            line)
        if m1:
            down_n, mask_n, pool_n, down_last = m1[1], m1[2], m1[3], m1[4]
            forward_result.append(
                f"    std::tuple<torch::Tensor, torch::Tensor> tmp_{pool_n} = {pool_n}->forward_with_indices({down_last});\n")
            forward_result.append(
                f'    torch::Tensor {down_n} = std::get<0>(tmp_{pool_n});\n')
            forward_result.append(
                f'    torch::Tensor {mask_n} = std::get<1>(tmp_{pool_n});\n')
        else:
            line = re.sub(r'self\.(\w+)', '\\1->forward', line)
            forward_result.append(f'    torch::Tensor {line};\n')
    elif line.strip() == '':
        forward_result.append('\n')
    else:
        forward_result.append('    //' + line.strip() + '\n')

# write outputs
output_file.write("""#pragma once

#include <torch/torch.h>

struct DeConvNetImpl : public torch::nn::Module {
  DeConvNetImpl() {
""")
output_file.writelines(init_result)
output_file.write("""}

torch::Tensor forward(torch::Tensor x) {
""")
output_file.writelines(forward_result)
output_file.write("""}
""")


for type, vars in var_names.items():
    output_file.write(f'  {type} {", ".join(x+"=nullptr" for x in vars)};\n')

output_file.write("""};

TORCH_MODULE(DeConvNet);
""")
output_file.close()
