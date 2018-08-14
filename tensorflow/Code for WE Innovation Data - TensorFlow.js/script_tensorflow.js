/**
* Async function To train a model using tensorflow.js
*/
async function train_tensor(training_data, target_data, MAXEPOCHES, TOTAL_SAMPLES, SOURCE_DIMENSIONS, TARGET_DIMENSIONS, FLAG_SAVE_MODEL) {

			//const training_data = tf.tensor2d([[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,0],[0,0,0,1,1],[0,0,1,0,0],[0,0,1,0,1],[0,0,1,1,0],[0,0,1,1,1],[0,1,0,0,0],[0,1,0,0,1],[0,1,0,1,0],[0,1,0,1,1],[0,1,1,0,0],[0,1,1,0,1],[0,1,1,1,0],[0,1,1,1,1],[1,0,0,0,0]]);
			//const target_data = tf.tensor2d([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]);
			//console.log("Training dataset is \n"+ training_data +"\n");	//error could be occurred here !
			//console.log("Target dataset is \n"+ target_data +"\n");	//error could be occurred here !

			//	1. Initialize the parameters necessary for training the model
			//----------------------------
			var arrLoss = [];	//array of error rates for every Epochs (training iterations)
			var arrIndividualCorrectRate = [];
			var arrSequenceCorrectRate = [];
			var arrEpoch = [];	//array of Epochs (training iterations)

			//var TOTAL_SAMPLES = 17;
			//var SOURCE_DIMENSIONS  = 5;	//input/source training data
			//var TARGET_DIMENSIONS = 4;	//output/target training data

			var i = 0;			//indicate each epoch in loop
			//var MAXEPOCHES = 20;	//number of maximum epochs to train the new model
			var MAXEPOCHES = MAXEPOCHES;	//number of maximum epochs to train the new model
			var BATCHSIZE = TOTAL_SAMPLES;	//17;	//back-propagation will be done for each batch after inputting #BATCHSIZE samples
													//value of Loss will be updated at every #BATCHSIZE samples have been input.

			console.log("MAX Epochs = "+MAXEPOCHES + ",\tTOTAL SAMPLES = "+TOTAL_SAMPLES+",\tSOURCE DIMENSIONS = "+SOURCE_DIMENSIONS+",\t TARGET DIMENSIONS = "+TARGET_DIMENSIONS);
			console.log("FLAG SAVE MODEL = "+FLAG_SAVE_MODEL);

			//==========================================================================
			//	2. 	Create the model/architecture of Tensorflow
			//==========================================================================

			//*
			const model = tf.sequential();

			model.add(tf.layers.dense({units: 100, activation: 'sigmoid', inputShape: [SOURCE_DIMENSIONS]}));	//first hidden layer
			//model.add(tf.layers.dense({units: 100, activation: 'sigmoid', inputShape: [100]}));
			model.add(tf.layers.dense({units: TARGET_DIMENSIONS , activation: 'sigmoid', inputShape: [100]}));	//second hidden layer

			model.compile({loss: 'meanSquaredError',   optimizer: 'rmsprop'});				//output layer
			//model.compile({loss: 'binaryCrossentropy', optimizer: 'sgd',   lr:0.1});

			//*/

			//=============================================================================================
			//	3. 	Training a new model using the architecture/structure defined in the previous part 2.
			//=============================================================================================
			//*
			//	3.1. Start observing the training time
			//------------------------------------
			var startDate = new Date(); // for now
			var startTime = startDate.getTime();
			var startTimeStr = startDate.getHours() +"h "+ startDate.getMinutes() +"mn "+ startDate.getSeconds()+"sec.";

			//	3.2. Start training the model
			//-----------------------------------------------------------------------------------------------------------------------
			//	 	This training is to observe the decrease of loss function "means squared errors". "arrLoss"
			//		In general, this value represents the percentage of loss/error rate of the model.
			//		It will be decreased after many training iterations (Epochs). After many Epochs,
			//		this value is trying to reach the value 0% of error.
			//			- when it reaches 0%, this means that the trained model is well trained.
			//			- in most case, the researcher don't want to have this 0% value because it is called over-fitting.
			//			  This means that the trained model is too fit into the training data (obsessed with only training data).
			//			  The trained model won't be a generic model which can work with other data (not in the training dataset).
			//-----------------------------------------------------------------------------------------------------------------------

			//*
			console.log("----------------------------------------------------------------");
			console.log("Training process has just started from now !\n\n");

			for (i = 1; i <= MAXEPOCHES ; ++i) {
				var h = await model.fit(training_data, target_data, {batchSize:BATCHSIZE, epochs: MAXEPOCHES, shuffle:true});



				var predicted_result = model.predict(training_data);
				predicted_result = tf.round(predicted_result);	//convert to integer

				//---compare the target and predicted results
				//----------------------------------------------
				var evaluation_resutl = predicted_result.equal(target_data);
				//console.log("\n After comparison, Evaluation results: \n"+ evaluation_resutl);	//target_data.equal(predicted_result).print();

				//---Evaluate the trained models
				//--------------------------------------------

				//---Eval-1: Determine the value of individual correct rate
				//-----------------------------------------------------------
				var reduce_sum_vertically   = tf.sum(evaluation_resutl, 1, false);		//reduce dimension by sum the 2D array row-by-row vertically
				var reduce_sum_horizontally = tf.sum(reduce_sum_vertically, 0, false);	//reduce dimension by sum the 2D array row-by-row horizontally (again)
				var sum_individual_all = (Array.from(reduce_sum_horizontally.dataSync()))[0];	//value at inde 0 of array 1D

				var individual_correct_rate = (sum_individual_all/(TOTAL_SAMPLES * TARGET_DIMENSIONS * 1.0)).toFixed(4);

				//console.log("reduce_sum_vertically = "+ reduce_sum_vertically);
				//console.log("reduce_sum_horizontally = "+ reduce_sum_horizontally);
				//console.log("individual_correct_rate = "+ individual_correct_rate);

				//---Eval-2: Determine the value of sequence correct rate
				//-----------------------------------------------------------
				var CONSTANT_tfones_1D = tf.ones([TOTAL_SAMPLES], 'float32');	//console.log(""+CONSTANT_arr1D);
				var CONSTANT_tf_scalar = tf.scalar(TARGET_DIMENSIONS);	//console.log(""+CONSTANT_arr1D_scalar);
				var CONSTANT_tf_correct_sum_each_sequence = tf.mul(CONSTANT_tfones_1D, CONSTANT_tf_scalar);
				//console.log("CONSTANT_tf_correct_sum_each_sequence = "+ CONSTANT_tf_correct_sum_each_sequence);

				var evaluation_seq_resutl = (tf.cast(reduce_sum_vertically, 'float32')).equal(CONSTANT_tf_correct_sum_each_sequence);
				//console.log("\n After comparison, Evaluation sequence results: \n"+ evaluation_seq_resutl);

				var reduce_sum_sequence_horizontally   = tf.sum(evaluation_seq_resutl, 0, false);

				var sum_sequence_all = (Array.from(reduce_sum_sequence_horizontally.dataSync()))[0];	//value at inde 0 of array 1D
				//console.log("sum_sequence_all ="+sum_sequence_all);
				var sequence_correct_rate = (sum_sequence_all/(TOTAL_SAMPLES * 1.0)).toFixed(4);

				//console.log("\t------------------------------------------------------------");
				console.log("\tEpoch " + i + ",\t Loss : " + (h.history.loss[0]).toFixed(6) + ",\t Indiv_Correct_Rate : " + individual_correct_rate  +" ("+ (individual_correct_rate * 100).toFixed(2)+"%),\t Sequence_Correct_Rate : " + sequence_correct_rate +" ("+ (sequence_correct_rate * 100).toFixed(2)+"%)");

				//console.log("\t------------------------------------------------------------\n");
				//console.log("\n The target data is \n"+ tf.round(target_data) +"\n");
				//console.log("\n The result is \n"+ predicted_result +"\n");
				//console.log("\n The predicted results after rounded is \n"+ tf.round(predicted_result) +"\n");

				arrEpoch.push(""+i);
				arrLoss.push(""+h.history.loss[0]);
				arrIndividualCorrectRate.push(""+individual_correct_rate * 100);
				arrSequenceCorrectRate.push(""+sequence_correct_rate * 100);

			}	//--- end training time after the loop

			console.log("\nTraining process has finished!");
			console.log("----------------------------------------------------------------\n");
			//*/

			//	3.3. End of the training, then calculate the training duration
			//------------------------------------------------------------------
			var endDate = new Date(); // for now
			var endTime = endDate.getTime();
			var endTimeStr = endDate.getHours() +"h "+ endDate.getMinutes() +"mn "+ endDate.getSeconds()+"sec.";
			var trainDuration = Math.round((endTime - startTime)/1000) +"sec."; //in sec.;


			//console.log(util.inspect(h, { maxArrayLength: null }));

			//	3.4. Test the previously trained model
			//------------------------------------------------------------------
			//model.predict(training_data).print();
			var predicted_result = model.predict(training_data);
			//console.log("\n The result is \n"+ predicted_result +"\n");
			predicted_result 	 = tf.round(predicted_result);	//convert to Interger

			//console.log("\n The target data is \n"+ tf.round(target_data) +"\n");
			console.log("\n The predicted results after rounded is \n"+ tf.round(predicted_result) +"\n");
			//console.log("\n After comparison \n"+ predicted_result.equal(target_data));	//target_data.equal(predicted_result).print();

			//	3.5. Save the trained model
			//---------------------------------------------------------------
			if (FLAG_SAVE_MODEL == true){
				const saveResult = await model.save('downloads://model-tensorFlow-'+(i-1)+'_epoch');	//to download
				//const saveResult = await model.save('localstorage://model-tensorFlow-1');
			}

			//===========================================================================================================================
			//	6. 	Print the chart to observe the decrease of loss function "means squared errors".
			//		In general, this value represents the percentage of loss/error rate of the model.
			//		It will be decreased after many training iterations (Epochs). After many Epochs,
			//		this value is trying to reach the value 0% of error.
			//			- when it reaches 0%, this means that the trained model is well trained.
			//			- in most case, the researcher don't want to have this 0% value because it is called over-fitting.
			//			  This means that the trained model is too fit into the training data (obsessed with only training data).
			//			  The trained model won't be a generic model which can work with other data (not in the training dataset).
			//============================================================================================================================
			//---Draw chart for display the loss function value (meanSquaredError)
			var arrDatasets1 = [
							{
								"label":"Loss (Error Rate)",
								"data":arrLoss,
								"fill":false,
								"borderColor":"rgb(75, 192, 192)",
								"lineTension":0.1
							}
						];
			drawChart("myChart-loss-vs-epochs", arrEpoch, arrDatasets1);

			//---Draw chart for display the correct rates
			var arrDatasets2 = [
							{
								"label":"Loss (Error Rate)",
								"data":arrIndividualCorrectRate,
								"fill":false,
								"borderColor":"rgb(75, 192, 192)",
								"lineTension":0.1
							},
							{
								"label":"Sequence Correct Rate (Error Rate)",
								"data":arrSequenceCorrectRate,
								"fill":false,
								"borderColor":"rgb(192, 192, 192)",
								"lineTension":0.1
							}
						];
			drawChart("myChart-correctRate-vs-epochs", arrEpoch, arrDatasets2);


			return {
				startTimeString: startTimeStr,
				endTimeString: endTimeStr,
				trainDurationString: trainDuration,
				predictedResult : predicted_result
			};

}




/**
 * Async function To train a model using tensorflow.js
 */
async function test_tensor(jsonUploadPath, weightsUpload, testing_tensor_data) {

			//-------------------------------
			// 1.	Load the trained model from the upload files
			//-------------------------------

			/*
			//const model = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json')
			//const modelLoad = await tf.loadModel('localstorage://model-tensorFlow-1');

			//const jsonUpload = document.getElementById('json-upload');
			//const weightsUpload = document.getElementById('weights-upload');

			//console.log(jsonUpload, weightsUpload);
			//console.log(tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));

			//1.1.	Load the tensorflow model (json and weights files) from the upload input form
			//------------------------------------------------------------------------------------------
			//const modelLoad = await tf.loadModel(tf.io.browserFiles([jsonUpload.files[0], weightsUpload.files[0]]));
			//*/


			//1.1.	Load the tensorflow model (json and weights files) from the http:// file
			//----------------------------------------------------------------------------------------
			const modelLoad = await tf.loadModel(jsonUploadPath);

			//1.2.	Predict the values based on the input data
			//----------------------------------------------------------------
			var predicted_result = modelLoad.predict(testing_tensor_data);

			//1.3.	Display the predicted results
			//-------------------------------------------
			//console.log("\n The result is \n"+ predicted_result +"\n");
			//console.log("\n The predicted results after rounded is \n"+ tf.round(predicted_result) +"\n");

			return predicted_result;

			//return {
			//	testing_tensor_data: testing_tensor_data,
			//	predictedResult : predicted_result
			//};


			//*/

}


/**
 *	Function to draw a graph/chart
 */
 function drawChart(tag_id, arrEpoch, arrDatasets){

	var ctx = document.getElementById(tag_id);
	var myChart = new Chart(ctx,{
		"type":"line",
		"data":{
			"labels":arrEpoch,
			"datasets": arrDatasets
		},
		"options":{
			maintainAspectRatio: true,
		}
	});
 }
