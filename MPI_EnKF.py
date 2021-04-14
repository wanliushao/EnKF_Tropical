#This program is MPI-EnKF algorithm
#Right now it just for sigle grid
#Update 08/03/2020 Shaoqing
import os
import glob
import numpy as np
import numpy.matlib
from mpi4py import MPI
from datetime import date
from netCDF4 import Dataset
import scipy.stats as stats
from numpy.linalg import inv

comm=MPI.COMM_WORLD
rank=comm.Get_rank()

recv_data = None
file1='/home/sqliu/software_install/cesm1_2_2/scripts/BR_Sa1'
file2='/BR_Sa1_trans_'
file3='/rundir'
command1='./BR_Sa1_trans_'
command2='.run'

EnKF_size=100
EnKF_number=11
npros=11
EnKF_time=24
per_jobs=EnKF_size/(npros-1); #the number of jobs for each processor to handle

if __name__=="__main__":
	def main():
		if (rank>0):
			for m in range(1,EnKF_time):
				subpro(rank);
				comm.Barrier();
		elif (rank==0):
			data=np.arange(npros)
			gpp_obs=np.empty(0);
			et_obs=np.empty(0);
			litter_obs=np.empty(0);
			for line in open('obs_fluxes.txt'):
				line=line.rstrip('\n');
				gpp_obs=np.append(lai_obs,line.split(" ")[0]);
				et_obs=np.append(lai_obs,line.split(" ")[1]);
				litter_obs=np.append(lai_obs,line.split(" ")[2]);
			print("process {} scatter data {} to other processes".format(rank, data))

			for m in range(1,EnKF_time+1):

				for j in range(1,EnKF_size+1):
					file=file1+file2+format(j, '003')
					os.chdir(file);

				for i in range(1,npros):
					comm.Send(data[i-1], dest=i, tag=11)
				comm.Barrier();
				for j in range(1,EnKF_size+1):
					file=file1+file2+format(j, '003')
					os.chdir(file)
					if (m==1):
						cmdstring='sed -i "79s/FALSE/TRUE/" env_run.xml'
						os.system(cmdstring)
				observation=np.matlib.zeros((3,1));
				observation[0,0]=gpp_obs[m-1];
				observation[1,0]=et_obs[m-1];
				observation[2,0]=litter_obs[m-1];
				observation=observation.astype(np.float);		

				Y_f=Get_simulation(EnKF_number, EnKF_size);
				logfid=open('/home/sqliu/software_install/cesm1_2_2/scripts/BR_Sa1/log','a');

				Y_update=EnKF(observation,Y_f,EnKF_size,EnKF_number)
				for k in range(EnKF_size):
					line=str(Y_update[0,k])+'\t'+str(Y_update[1,k])+'\t'+str(Y_update[2,k])+'\t'+str(observation[0,0])+'\t'+str(observation[1,0])+'\t'+str(observation[2,0])+'\n';
					logfid.write(line)
				logfid.close();

				Update(Y_update, EnKF_size)				

		MPI.Finalize()

	def subpro(rank):
		data=np.empty(1);
		comm.Recv(data, source=0, tag=11)
		for i in range(1,per_jobs+1):
			indx=(rank-1)*per_jobs+i;
			file=file1+file2+format(indx, '003')
			print("process {} deal with directory{}".format(rank, file))
			os.chdir(file)
			cmdstring=command1+format(indx, '003')+command2
			os.system(cmdstring)

	def Get_simulation(EnKF_number,EnKF_size):
		Y_f=np.matlib.zeros((EnKF_number,EnKF_size))
		for j in range(1,EnKF_size+1):
			file=file1+file2+format(j, '003')+file3
			os.chdir(file)
			newest = max(glob.iglob('*clm2.h0.*nc'), key=os.path.getctime)
			ncfid=Dataset(newest,'r')
			GPP=ncfid.variables['GPP']
			ET=ncfid.variables['FCEV']+ncfid.variables['FCTR']
			Litter=ncfid.variables['LEAFC_TO_LITTER	']
			ncfid.close()
			Y_f[8,j-1]=GPP;
			Y_f[9,j-1]=ET;
			Y_f[10,j-1]=Litter;
			pft_file='/home/sqliu//software_install/cesm1_2_2/Mojave_input/lnd/clm2/pftdata/pft-physiology_constant_allocation_'+str(j)+'.nc'
			para=['Vcmax25', 'leafcn', 'rootb_par', 'z0mr', 'froot_leaf', 'frootcn', 'mp_pft', 'leaf_long']
			ncfid=Dataset(pft_file,'r')
			for k in range(8):
				Y_f[k,j-1]=ncfid.variables[para[k]][4]
			ncfid.close();
			
		return (Y_f)

	def EnKF(observation,simulation, EnKF_size, EnKF_number):
		observation_number=3;
		H=np.matlib.zeros((observation_number,EnKF_number));
		H[0,3]=1;
		H[1,4]=1;
		H[2,5]=1;
		R=np.matlib.zeros((observation_number,observation_number));
		np.fill_diagonal(R,np.square(error));
		f_mean=np.asmatrix(np.mean(simulation,axis=1));
		Ensemble_dev=simulation-np.repeat(f_mean,EnKF_size,axis=1);
		Pb=np.dot(Ensemble_dev,np.transpose(Ensemble_dev))/(EnKF_size-1);
		temp1=np.dot(np.dot(H,Pb),np.transpose(H))+ R;
		temp2=np.dot(Pb,np.transpose(H));
		K=np.dot(temp2,inv(temp1));
		u_mean=f_mean + np.dot(K,(observation-np.dot(H,f_mean)));

		temp1=1/(np.sqrt(temp1)) 
		K_p=np.dot(temp2,np.transpose(temp1))
		K_p=np.dot(K_p,inv(inv(temp1)+np.sqrt(R)))
		obs_perturb=np.zeros((1,EnKF_size))
		temp1=simulation-np.repeat(f_mean,EnKF_size,axis=1);
		EnKF_perturb=temp1+np.dot(K_p,(obs_perturb-np.dot(H,temp1)))
		Y_update=np.repeat(u_mean,EnKF_size,axis=1)+EnKF_perturb
		return (Y_update)

	def Update(updates,EnKF_size):
		para=['Vcmax25', 'leafcn', 'rootb_par', 'flnr', 'froot_leaf', 'frootcn', 'mp_pft', 'leaf_long']
		stdvs=[10,20,1.5,0.025,0.5,20,5,0.5]
		for j in range(1,EnKF_size+1):
			pft_file='/home/sqliu//software_install/cesm1_2_2/Mojave_input/lnd/clm2/pftdata/pft-physiology_constant_allocation_'+str(j)+'.nc'
			ncfid=Dataset(pft_file,'r+');
			for k in range(8):
				ncfid.variables[para[k]][4]=0.99*updates[k,j-1]+0.1*stdvs[k]
			ncfid.close();

	main();
