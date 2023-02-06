#!/usr/bin/bash
source /home/macierz/s175454/tomography_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/macierz/s175454/tomography

nohup $( \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=8 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=8 --epochs=50 --gpu=1 --fold=1 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=8 --epochs=50 --gpu=1 --fold=2 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=8 --epochs=50 --gpu=1 --fold=3 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=8 --epochs=50 --gpu=1 --fold=4 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=1 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=1 --fold=1 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet \
       ) &

nohup $( \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=2 --fold=2 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=2 --fold=3 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=2 --fold=4 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=quanet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=2 --fold=0 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=2 --fold=1 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=2 --fold=2 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus \
       ) &

nohup $( \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=3 --fold=3 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=4 --epochs=50 --gpu=3 --fold=4 --learning_rate=0.0001 --img_size=512 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unetplusplus; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=16 --epochs=50 --gpu=3 --fold=0 --learning_rate=0.0001 --img_size=256--metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=16 --epochs=50 --gpu=3 --fold=1 --learning_rate=0.0001 --img_size=256--metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=3 --fold=2 --learning_rate=0.0001 --img_size=256--metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=3 --fold=3 --learning_rate=0.0001 --img_size=256--metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet \
      ) &

nohup $( \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=4 --fold=4 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=4 --fold=0 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=2 --epochs=50 --gpu=4 --fold=1 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=defednet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=16 --epochs=50 --gpu=4 --fold=2 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=16 --epochs=50 --gpu=4 --fold=3 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet; \
        python ./src/run_training.py --normalize --multiclass --discard --batch_size=16 --epochs=50 --gpu=4 --fold=4 --learning_rate=0.0001 --img_size=256 --metadata="/home/macierz/s175454/lits_prepared/metadata.csv" --dataset="/home/macierz/s175454/lits_prepared" --net_name=unet \
       ) &
