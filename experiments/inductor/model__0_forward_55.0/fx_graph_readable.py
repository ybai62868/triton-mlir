class GraphModule(torch.nn.Module):
    def forward(self, primals_1: f32[1, 3, 224, 224], primals_2: f32[1, 3, 224, 224]):
        # File: conv.py:12, code: x = x + torch.ones_like(x)
        full_default: f32[1, 3, 224, 224] = torch.ops.aten.full.default([1, 3, 224, 224], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        add: f32[1, 3, 224, 224] = torch.ops.aten.add.Tensor(primals_2, full_default);  primals_2 = full_default = None
        
        # File: conv.py:13, code: res = torch.nn.functional.conv2d(x, self.weight)
        convolution: f32[1, 1, 1, 1] = torch.ops.aten.convolution.default(add, primals_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        
        # File: conv.py:14, code: res = res + torch.ones_like(res)
        full_default_1: f32[1, 1, 1, 1] = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        add_1: f32[1, 1, 1, 1] = torch.ops.aten.add.Tensor(convolution, full_default_1);  convolution = full_default_1 = None
        return [add_1, primals_1, add]
        
