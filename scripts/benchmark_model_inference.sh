# bin/bash -l
conda activate /data/stripeli/condaenvmlbench

#echo "BrainAge"
#deepsparse.benchmark -b 1 -t 60 -w 10 --export_path /tmp/brainage_cnn.0percent.inference_eval.json /tmp/brainage_cnn.0percent.onnx
#deepsparse.benchmark -b 1 -t 60 -w 10 --export_path /tmp/brainage_cnn.85percent.inference_eval.json /tmp/brainage_cnn.85percent.onnx
#deepsparse.benchmark -b 1 -t 60 -w 10 --export_path /tmp/brainage_cnn.90percent.inference_eval.json /tmp/brainage_cnn.90percent.onnx
#deepsparse.benchmark -b 1 -t 60 -w 10 --export_path /tmp/brainage_cnn.95percent.inference_eval.json /tmp/brainage_cnn.95percent.onnx
#deepsparse.benchmark -b 1 -t 60 -w 10 --export_path /tmp/brainage_cnn.95percent.inference_eval.json /tmp/brainage_cnn.99percent.onnx
#
#echo "FashionMNIST"
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/fashionmnist_fc.0percent.inference_eval.json /tmp/fashionmnist_fc.0percent.onnx
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/fashionmnist_fc.80percent.inference_eval.json /tmp/fashionmnist_fc.80percent.onnx
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/fashionmnist_fc.85percent.inference_eval.json /tmp/fashionmnist_fc.85percent.onnx
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/fashionmnist_fc.90percent.inference_eval.json /tmp/fashionmnist_fc.90percent.onnx
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/fashionmnist_fc.95percent.inference_eval.json /tmp/fashionmnist_fc.95percent.onnx
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/fashionmnist_fc.99percent.inference_eval.json /tmp/fashionmnist_fc.99percent.onnx
#
#
#echo "CIFAR-10 CNN"
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_cnn.0percent.inference_eval.json /tmp/cifar10_cnn.0percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_cnn.80percent.inference_eval.json /tmp/cifar10_cnn.80percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_cnn.85percent.inference_eval.json /tmp/cifar10_cnn.85percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_cnn.90percent.inference_eval.json /tmp/cifar10_cnn.90percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_cnn.95percent.inference_eval.json /tmp/cifar10_cnn.95percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_cnn.99percent.inference_eval.json /tmp/cifar10_cnn.99percent.onnx
#
#
#echo "CIFAR-10 ResNet"
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.0percent.inference_eval.json /tmp/cifar10_resnet.0percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.80percent.inference_eval.json /tmp/cifar10_resnet.80percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.85percent.inference_eval.json /tmp/cifar10_resnet.85percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.90percent.inference_eval.json /tmp/cifar10_resnet.90percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.95percent.inference_eval.json /tmp/cifar10_resnet.95percent.onnx
#deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.99percent.inference_eval.json /tmp/cifar10_resnet.99percent.onnx
#
#
echo "CIFAR-100 VGG-16"
deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.0percent.inference_eval.json /tmp/cifar100.vgg16.0percent.onnx
deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.90percent.inference_eval.json /tmp/cifar100.vgg16.90percent.onnx
deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.95percent.inference_eval.json /tmp/cifar100.vgg16.95percent.onnx
deepsparse.benchmark -b 128 -t 60 -w 10 --export_path /tmp/cifar10_resnet.99percent.inference_eval.json /tmp/cifar100.vgg16.99percent.onnx


#echo "IMDB Bi-LSTM"
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/imdb_bilstm.0percent.inference_eval.json /tmp/imdb_bilstm.0percent.onnx
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/imdb_bilstm.85percent.inference_eval.json /tmp/imdb_bilstm.85percent.onnx
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/imdb_bilstm.90percent.inference_eval.json /tmp/imdb_bilstm.90percent.onnx
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/imdb_bilstm.95percent.inference_eval.json /tmp/imdb_bilstm.95percent.onnx
#deepsparse.benchmark -b 32 -t 60 -w 10 --export_path /tmp/imdb_bilstm.99percent.inference_eval.json /tmp/imdb_bilstm.99percent.onnx
