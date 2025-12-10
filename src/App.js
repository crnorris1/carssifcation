import logo from './logo.svg';
import './App.css';
import React from 'react'

function App() {

  const [image, setImage] = React.useState(null)

  const onFileChange = (event) => {
    setImage(event.target.files[0])
  }

  function submitImage(){
    const formData = new FormData()
    formData.append(
      "image",
      image,
      image.name
    )
    alert(image.name)

  }

  return (
    <div className="App">
      <h1>Welcome to Car Classifier</h1>
      <h3>CS4342 Final Project - Cam Norris, Akaash Walker</h3>
      <h2>Upload a picture of your car <b>from the side </b> and find out what type of car it is</h2>
      <h2>(SUV, Sudan, Coupe, Truck)</h2>
      <br/><br/>
      <form>
        <h2>Upload your photo:</h2>
        <input type="file" id="imageUpload" onChange ={onFileChange}></input>
        <button onClick={submitImage}>Upload</button>
      </form>
    </div>
  );
}

export default App;
