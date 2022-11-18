import torch
from torch.autograd import Variable


#------------------查看bottom给top时, 传递(yb+10)  和传递 yb_new=(yb+10)对 梯度的影响
# 法一传递yb_new：
wb = torch.tensor(2.0)
wb = Variable(wb, requires_grad=True)
wt = torch.tensor(4.0)
wt = Variable(wt, requires_grad=True)

x = torch.tensor(3)
yb = wb*x
yb_new = (yb+10).clone().detach().requires_grad_(True)
yt = wt*yb_new


yt.backward()
print('wt.grad', wt.grad)
print('yb_new.grad', yb_new.grad)
print('yb.grad', yb.grad)
print('wb.grad', wb.grad)


# 法二--传递(yb2+10)
wb2 = torch.tensor(2.0)
wb2 = Variable(wb2, requires_grad=True)
wt2 = torch.tensor(4.0)
wt2 = Variable(wt2, requires_grad=True)

x2 = torch.tensor(3)
yb2 = wb2*x2
yb2 = Variable(yb2, requires_grad=True)
yt2 = wt2*(yb2+10)

yt2.backward()

print('wt2.grad', wt2.grad)
print('yb2.grad', yb2.grad)
print('wb.grad', wb.grad)


# 输出结果:
# wt.grad tensor(16.)
# yb_new.grad tensor(4.)
# yb.grad None
# wb.grad None

# wt2.grad tensor(16.)
# yb2.grad tensor(4.)
# wb.grad None

#------结论:分开给top还是一起给top, 返回梯度一样
#---------------------------------------------------------------

#------------------查看bottom.backward的时候, (yb_new).backward()与(yb+10).backward()的区别
# 法一---(yb+10).backward()：
wb = torch.tensor(2.0)
wb = Variable(wb, requires_grad=True)
wt = torch.tensor(4.0)
wt = Variable(wt, requires_grad=True)

x = torch.tensor(3)
yb = wb*x
yt = wt*(yb+10)
(yb+10).backward()
print('wb.grad', wb.grad)


# 法二---yb_new.backward():
wb2 = torch.tensor(2.0)
wb2 = Variable(wb2, requires_grad=True)
wt2 = torch.tensor(4.0)
wt2 = Variable(wt2, requires_grad=True)

x2 = torch.tensor(3)
yb2 = wb2*x2
yb2 = (yb2+10).clone().detach().requires_grad_(True)

yt = wt2*yb_new
yb2.backward()
print('wb2.grad', wb2.grad)


# 输出结果:
# wb.grad tensor(3.)
# wb2.grad None

#------结论:(yb+10).backward模型会更新, yb_new.backward()底层模型不更新
#----------------------------------------------------------------------------




