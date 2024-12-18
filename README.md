working files:
`cluster.yaml`
`distributed-training-2.py`
`Dockerfile2`
`pytorch-job-cpu.yaml` 


https://vmware.github.io/vSphere-machine-learning-extension/user-guide/training-pytorchjob.html

## STEP 1 - create IAM user

The first step is to create a user with admin permissions and keys to programmatically acces aws resources (this is is required to use `eksctl create`).

setup up the user with the credentials with the aws cli using `aws configure`.

## STEP 2 - create a bucket to save trained model

Then we create an s3 bucket using the aws cli. This bucket will be used to store the trained model.

`aws s3 mb s3://pytorch-svw --region us-east-1`

The bucket name is used in the `cluster.yaml` that will be used to create the cluster from the manifiest. 

## STEP 3 - create a cluster

The first step is to create a cluster using the manifest in the `eks-cluster` directory. Note that there are also some other example cluster manifest that can be looked in. 

`eksctl create cluster -f eks-cluster/cluster.yaml`

Please note that this manifest also creates a managed nodegroup. In the nodegroup m5.large are used. These are the biggest instances I can get in the AWS sandbox that I am using. The manifest also creates a oidc provider and a service account `pytorch-training-sa` for S3 access to the bucket we created above. The creation of the cluster takes about 15-20 minutes.

When the cluster is finished you can see this in the EKS service or the cloudformation stack(s) used to create the cluster. 

```
# if you get the cluster returned you are good to go
eksctl get cluster
```

```
# to get the nodes 
kubectl get nodes
```

## STEP 4 - install the kubeflow operator

After the cluster is created use the following command to install the kubeflow training operator

`
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.5.0"`
`

run 

`
kubectl get crd
`

this should show that we have `pytorchjobs.kubeflow.org` installed

check that the training operator is running via:

`kubectl get pods -n kubeflow`

this should show that we have the `training-operator-xxxx` running.

## STEP 5 - prepare distributed training script

A working version of training using pytorch is available in `distributed-train.py` that is setup for distributed training. 

There is a video series on pytorch that explains changes to a normal training script in order to train a model in a distributed fashion. 

Especially leveraging torchrun simplifies the steps towards distributed training.

train on local machine

`torchrun --standalone --nnodes=1 --nproc-per-node 2 distributed-training-2.py`

This starts training on one node (our local machine) with 2 processes on this node.

Please note that this script was setup to train on cpu's. It uses the 'gloo' backend for CPU-based distributed training. This was done to avoid unnessary costs by spinning up expensive GPU instances. If you want to use GPU instances you have to bring the model and data to the GPU using the local rank. 


## STEP 6 - Dockerize the training script.

See the `Dockerfile` used to create an image that has pytorch and boto3 installed. Note that also the train and test data is copied in the image. Normally you want to access data remotely (e.g. s3, EBS, EFS). For now it was done to just have a smoother operation. 

build the docker image using:

`docker build -t longtong/pytorch-cpu .`

if the images has not the username included use `docker tag <image-name> <username>/<image-name>`.

Test the container locally

`docker run --rm longtong/cpu`

In order to use the image in kubernetes we need to push it to a central repositry.

`docker push longtong/pytorch-cpu`

now the image is ready to be pulled from docker hub on our eks cluster. 

## STEP 6 - run distributed training on EKS cluster

It took me quite some time to get the training script and the pytorch job working correclty. But now we are ready to execute the training job on our eks cluster:

`kubectl apply -f pytorch-job-cpu.yaml`

monitor the training:

`kubectl get events --sort-by='.metadata.creationTimestamp'`

check if the correct images is used.

`kubectl describe pod pytorchjob-distributed-training-master-0 | grep "Image:"`


```
# check job status
kubectl get pytorchjob
```

```
# check status of pods
kubectl get po
```

```
# Watch both pods -w means live tracking
kubectl get pods -w
```

```
# get more details
kubectl describe pytorchjob pytorch-training
```

```
# check logs for master
kubectl logs -f pytorch-training-master-0
```

```
# check logs for worker
kubectl logs -f pytorch-training-worker-0
```

The code in `distributed-training.py` runs.
`torchrun --nproc_per_node=2 distributed-training.py`


### Test connectiviy

Create a test pod

`kubectl apply -f debug-pod.yaml`

# Get into the pod
kubectl exec -it network-debug -- sh

# Once inside, you can run various network tests:
# Test DNS resolution
nslookup pytorchjob-distributed-training-master-0

# Try pinging the IP address directly
ping 192.168.35.254

# Ping the master pod
ping pytorchjob-distributed-training-master-0

My experience is that sometimes the service discovery might take a while. Check the logs of the pod to see if they have started training and only then execute the commands below. 

# Test if port is accessible
nc -zv pytorchjob-distributed-training-master-0 29500

nc -zv pytorchjob-distributed-training-master-0.default.svc.cluster.local 29500

# Check all pods in the namespace
wget -qO- http://pytorchjob-distributed-training-master-0:29500


You can watch the pods come up with:

kubectl get pods -w

Check job status with:

kubectl get pytorchjob
For logs from all pods:

kubectl logs -l pytorch-job-name=pytorchjob-distributed-training

If something goes wrong, describe the job for more details:

kubectl describe pytorchjob pytorchjob-distributed-training

#### [OPTIONAL] create and attach iam policy to serive account

In the cluster.yaml that was used to already all steps are included in this section. The steps in this section can be used if your cluster has not created the service account in the cluster.yaml.

```
# Create IAM policy
aws iam create-policy \
    --policy-name pytorch-s3-policy \
    --policy-document file://eks-cluster/s3-policy.json
```

```
eksctl utils associate-iam-oidc-provider --region=us-east-1 --cluster=pytorch-training-cluster
```

```
# Create IAM role for service account
eksctl create iamserviceaccount \
    --name pytorch-training-sa \
    --namespace default \
    --cluster pytorch-training-cluster \
    --attach-policy-arn arn:aws:iam::905418242537:policy/pytorch-s3-policy \
    --approve \
    --override-existing-serviceaccounts
```

```
kubectl get serviceaccounts
```


