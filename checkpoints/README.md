# PrecisionTrack checkpoints and hyperparameters guide

PrecisionTrack's [train_detection.py](https://github.com/VincentCoulombe/precision_track/tree/main/tools) tool will output some obscure and (I am sure) uninteresting files into your defined [deploying_directory](https://github.com/VincentCoulombe/precision_track/tree/main/configs) at the end of its run. **THESE FILES ARE VERY IMPORTANT**, they **are** your optimized PrecisionTracker.

That said, the following guide is intended to help you better understand what these files are, contain and how they are used in practice.

---

## 1) The `.pth` file

The only file that is certain to appear inside your [deploying_directory](https://github.com/VincentCoulombe/precision_track/tree/main/configs) once the [train_detection.py](https://github.com/VincentCoulombe/precision_track/tree/main/tools) successfully ran is going to be your `.pth` file. This file's name will look something like `model_<your dataset name>_DEPLOYED.pth`.

This file:

- **Is**: A [Pytorch checkpoint](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
- **Contains**: The optimized **weights** of your PrecisionTrack's detection network (All the knowledge it learned during its training).
- **Is usefull for**: When you will use PrecisionTrack to track on new videos, the system will _upload_ these weights so it has the knowledge to perform accurate detection, species classification and pose-estimation on your data.

## 2) The `.onnx` file

This file will only appear if the [train_detection.py](https://github.com/VincentCoulombe/precision_track/tree/main/tools)'s `--deploy` parameter is set to `==true` (it is by default). The file's name will look something like `model_<your dataset name>_DEPLOYED.onnx`.

This file:

- **Is**: An [ONNX checkpoint](https://fr.wikipedia.org/wiki/Open_Neural_Network_Exchange).
- **Contains**: The same thing as your `.pth` file, but formatted differently.
- **Is usefull for**: Running a `.onnx` tracker is about 50% faster than running a `.pth` file. Therefore, it has a higher [loading priority](https://github.com/VincentCoulombe/precision_track/tree/main/configs) than the `.pth`.

## 3) The `.engine` file

This file will only appear if the [train_detection.py](https://github.com/VincentCoulombe/precision_track/tree/main/tools)'s `--deploy` parameter is set to `==true` (it is by default) **AND** if your machine is CUDA-accelerated. The file's name will look something like `model_<your dataset name>_DEPLOYED_<your gpu name>.engine`.

This file:

- **Is**: A [TensorRT checkpoint](https://www.geeksforgeeks.org/deep-learning/what-is-tensorrt/).
- **Contains**: The same thing as your `.pth` file, but formatted differently.
- **Is usefull for**: Running a `.engine` tracker (on the GPU its was optimized for) is about 50% faster than running a `.onnx` file. Therefore, it has the highest [loading priority](https://github.com/VincentCoulombe/precision_track/tree/main/configs).

---

## 4) The `hyperparameters.json` file

The `hyperparameters.json` contains (as its name suggests) your tracking system's hyperparameters. knowing its content and how it impact your PrecisionTracker **is highly recommended** as most tracking _bugs_ could be fixed by tuning these hyperparameters.

That said, the content of that file should look something like this:

```json
{
	"calibrated_temperature": 0.6,
	"tracking_thresholds": {
		"init_thr": 0.8,
		"conf_thr": 0.55,
		"low_thr": 0.05
	},
	"stitching_hyperparams": {
		"beta": 0.25,
		"match_thr": 0.8,
		"eps": 0.01
	}
}
```

As you can see, it contains three main keys:

- calibrated_temperature
- tracking_thresholds
- stitching_hyperparams

Whose roles are explained in the following sub-sections...

### calibrated_temperature

This hyperparameters is used to calibrate your detection model so that the confidence level it outputs better reflect the expected accuracy level. It is explained in more detailed in the [manuscript](https://www.biorxiv.org/content/10.1101/2024.12.26.630112v3). Therefore, I recommend curious users to read it to better understand what it concretely does.

**Note**: Passing `--calibrate==true` (or nothing, since it defaults to true anyway...) to the `train_detection` tool will automatically load an optimal (or close to optimal) calibrated_temperature in your `hyperparameters.json` file.

### tracking_thresholds

These hyperparameters directly control how the [tracking behaves](https://www.biorxiv.org/content/10.1101/2024.12.26.630112v3). It is composed of three keys:

- init_thr
- conf_thr
- low_thr

Which correspond to thresholds the detection confidence scores must exceed in order to:

- Initiate a new track
- Be confident enough that the detection is a true positive to bypasses most "safety checks"
- Be confident enough in the detection to consider it as a potential true positive

So, you can play with them and see what combinaison better fit your tracking scenario.

For exemple, is you see a lot of false positive detections being tracked (duplications of the same entity, reflections, etc...), then it might be a good idea to increase your init_thr. this will tell your PrecisionTracker to only initialize tracks when you are **very confident** it is a true positive.

**Note**: Setting `display_untracked_detections` to `true` in your `user_configs.yaml` file will enable you to visualize your detections as well as your tracking. Therefore, we highly recommend setting it to `true` when tuning your hyperparameters in order to be able to see the concrete changes your modifications made.

### stitching_hyperparams

**Important**: These hyperparameters are only important **if** you have a predefined `num_subjects` in your `user_configs.yaml`. Please refer to our [configs guide](https://github.com/VincentCoulombe/precision_track/tree/main/configs) for more details.

These hyperparameters directly control how the [stitching algorithm behaves](https://www.biorxiv.org/content/10.1101/2024.12.26.630112v3). It is composed of three keys:

- beta
- match_thr
- eps

In most cases, do not change `eps` (it is mostly used for numerical stability). The other two impacts:

- How big the search zones will be (a smaller value will lead to bigger search zones)
- How loose will the algorithm stitch new tracks (a very high value (e.g. 0.99) will stitch almost every new tracks and a small value (e.g. 0.5) will only stitch tracks very close to a search zone)

For exemple, if you only have two subjects in your vivarium, you might want a much lower `beta` (e.g 0.1) and a much higher `match_thr` (e.g. 0.99) so that your subject's trajectories are automatically stitched together (you do not care about robustness since you know there is only 2 subjects in your vivarium). On the opposite, if you are tracking large cohorts, you might prefer a lower `match_thr` (e.g. 0.8) and a `beta` that better fit the size of your camera's field-of-view's hiding spots.

**Note**: Setting `display_search_zones` to `true` in your `user_configs.yaml` file will enable you to visualize your search zones. Therefore, we highly recommend setting it to `true` when tuning your hyperparameters in order to be able to see the concrete changes your modifications made.
