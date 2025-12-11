
import React from 'react'
import "./style.css"

function App() {

  const [image, setImage] = React.useState(null)
  const [uploaded, setUploaded] = React.useState(false)
  const [previewUrl, setPreviewUrl] = React.useState(null);

  const [fetching, setFetching] = React.useState(false)
  const [completed, setCompleted] = React.useState(false)

  const [prediction, setPrediction] = React.useState("")

  const fileInputRef = React.useRef(null);

  //Fires whenever a new file is uploaded
  const onFileChange = (event) => {

    //Reset values
    setCompleted(false)
    setFetching(false)
    setPrediction("")

    //Image file 
    const file = event.target.files[0]
    if (file){
      setImage(file)

      //Read the file for displaying
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreviewUrl(reader.result)
      }

      reader.readAsDataURL(file)
    }
    else{
      setImage(null)
      setPreviewUrl(null)
    }
    
  }

  //Fires when image is submitted
  async function submitImage(e){

    //Prevents page reload
    if (e) e.preventDefault()
    
    if (image){
      //Image is uploaded
      setUploaded(true)

      //The response from the backend server
      const response = await new Promise((resolve, reject) => {
        const reader = new FileReader()
        
        //When the reader is given an image
        reader.onload = async function (event){
          try{
            //Encodes the image into base 64 to be sent to the backend
            const base64String = event.target.result
            const encodedFile = String(base64String).split(',')[1]

            //Event to send to backend
            const send = {
              fileName : image.name,
              fileType : image.type,
              fileData : encodedFile,
            }
            
            setFetching(true)

            //Fetches 5000 server
            const result = await fetch('http://localhost:5000/sendImage', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json' 
              }, 
              body: JSON.stringify(send)
            })

            setFetching(false)
            setCompleted(true)

            //JSON'd result
            const resp5000 = await result.json()
            resolve(resp5000)
          }
          catch(error){
            alert(error)
            reject(error)
          }
        }
        reader.readAsDataURL(image)
      })

      // Reset file input's value after the upload completes so selecting the same file again triggers onChange
      if (fileInputRef.current) fileInputRef.current.value = '';

      //If a response was sent from backend
      if (response){
        //Grammar lol
        const pred = response.prediction
        var article = "a "
        if (pred === "suv")
          article = "an "

        //Prediction string
        setPrediction(article + response.prediction)
      }
      else{
        alert("No response")
      }
    }
    else{
      alert("Choose a file to upload.")
    }
  }

  return (
    <div className ="main">
      <h1>Car Classifier</h1>
      <h4>CS4342 Final Project - Cam Norris, Akaash Walker</h4>
      <h3>Upload a picture of a car <i className="emphasis">from the side</i> and find out what type of car it is!</h3>
      <h3>Categories: (SUV, Sudan, Coupe, Truck, Semi)</h3>
      <form>
        <h2>Upload your photo:</h2>
        <label className = "customUpload">
          Choose Image
          <input ref={fileInputRef} type="file" id="imageUpload" onChange ={onFileChange}></input>
        </label>
        <br/><br/>
        <button onClick={submitImage}>Upload</button>
        <br/><br/>
        {/* Show preview whenever we have one (and not while fetching). Also show prediction when completed */}
        {(previewUrl && !fetching) ? (
          <div>
            <img src = {previewUrl} className="output"></img>
            {completed ? (<h3>This is {prediction}</h3>) : null}
          </div>
        ) : (fetching) ? 
          (<h3>Loading...</h3>) : 
          (completed) ? 
            (<h3>This is {prediction}</h3>) : 
            <></>}
      </form>
        
    </div>
  );
}

export default App;
