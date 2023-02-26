#!/usr/bin/bash
source /home/macierz/s175454/tomography_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/macierz/s175454/tomography


nohup $( \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --multiclass --discard --batch_size=32 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=32 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --tumor --discard --batch_size=2 --epochs=50 --gpu=0 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=mix --seed=31 &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --tumor --discard --batch_size=2 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=mix --seed=32 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --multiclass --discard --batch_size=32 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=31 &>> global.log \

      ) &

nohup $( \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --multiclass --discard --batch_size=8 --epochs=50 --gpu=2 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=32 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --use_polar --multiclass --discard --batch_size=8 --epochs=50 --gpu=2 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=32 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --discard --batch_size=32 --epochs=50 --gpu=2 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=30 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --discard --batch_size=32 --epochs=50 --gpu=2 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=31 &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --tumor --discard --batch_size=32 --epochs=50 --gpu=43 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=32 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --discard --batch_size=32 --epochs=50 --gpu=3 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=32 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --discard --batch_size=2 --epochs=50 --gpu=3 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=mix --seed=31 &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --multiclass --discard --batch_size=32 --epochs=50 --gpu=4 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=30 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --multiclass --discard --batch_size=2 --epochs=50 --gpu=4 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=mix --seed=30 &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --tumor --discard --batch_size=32 --epochs=50 --gpu=5 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=30 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --tumor --discard --batch_size=2 --epochs=50 --gpu=5 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=mix --seed=30 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --tumor --discard --batch_size=32 --epochs=50 --gpu=5 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet --loss_name=mix --seed=31 &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --multiclass --discard --batch_size=2 --epochs=50 --gpu=6 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=mix --seed=32 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --discard --batch_size=2 --epochs=50 --gpu=6 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=mix --seed=30 &>> global.log \
      ) &

nohup $( \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --discard --batch_size=2 --epochs=50 --gpu=7 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=mix --seed=32 &>> global.log; \
        python ./src/run_training.py --no_val --ww=225 --wl=25 --multiclass --discard --batch_size=2 --epochs=50 --gpu=7 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet --loss_name=mix --seed=31 &>> global.log \
     ) &
