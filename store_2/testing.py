from  numpy import load

#data=load('trial_3_M_40_N_5_L_40_SNR_90.0_Tau_1_set_2.npz',allow_pickle=True)
data=load('train_loss_acc_and_test_loss_acc_and_training_loss_acc.npz',allow_pickle=True)
lst=data.files

for item in lst:
    print(item)
    print("###################")
    print(data[item])
    print("####################")