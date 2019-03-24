# Setting up SSH on Google Cloud

### Adding SSH Keys

Much of the documentation regarding adding and removing ssh keys is here:

https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys

Modify the public key file so that it has the format ssh-rsa [KEY_VALUE] [USERNAME].

Then, you can add the ssh key to the gcloud image through the GCP Console, in the metadata page. Copy the public ssh key file to the text box inside.



### Setting up the IP address

Now, we must set up an IP address; information on that is located here.

https://cloud.google.com/compute/docs/instances/connecting-advanced#thirdpartytools

If the gcloud instance already has an external IP address, then we can connect to it directly; finding this can be done on the Instances page.

However, if the instance does not have an external IP address, then the instance only has an internal IP address on a VPC network, which we can connect to through a VPN or a bastion host instance.

If we have a VPN that connects from the local network to the VPC, then we can connect through ssh using the internal instance IP address.

However, if we do not, then we must connect through bastion host; the information about this can be found in the below link.

https://cloud.google.com/compute/docs/instances/connecting-advanced#bastion_host

* Find the external IP address of the Linux bastion host instance, and find the internal IP address of the internal instance that you want to connect to. You can find the addresses in the **External IP** and **Internal IP** columns on your Instances page. 

* Connect to the Linux bastion host instance using either `ssh` or `gcloud compute ssh`. For either option, include the `-A` argument to enable authentication agent forwarding.

* Connect to the Linux bastion host instance and forward your private keys with `ssh`.

```
ssh -A [USERNAME]@[BASTION_HOST_EXTERNAL_IP_ADDRESS]
```

* Alternatively, you can connect to the bastion host instance and forward your private keys using the `gcloud compute ssh` command. This option allows you to connect to the bastion host instance using the `gcloud` command-line tool and then use regular `ssh` with the forwarded credentials when you connect to internal IP addresses.

```sh
gcloud compute ssh --ssh-flag="-A" [BASTION_HOST_INSTANCE_NAME]
```

After this, using ssh on a machine using Linux to connect to the gcloud instance should work.

From here, using the scp command to directly transfer files from the remote gcloud server to the other remote server, whether it be Openstack, Farnam, or something else, should work.

