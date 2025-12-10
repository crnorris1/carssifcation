
import React from 'react'
import axios from 'axios'

function App() {

  const [image, setImage] = React.useState(null)

  const onFileChange = (event) => {
    setImage(event.target.files[0])
  }

  async function submitImage(e){

    if (e) e.preventDefault()

    if (image){

      const response = await new Promise((resolve, reject) => {
        const reader = new FileReader()
        
        reader.onload = async function (event){
          
          try{
            
            const base64String = event.target.result
            const encodedFile = String(base64String).split(',')[1]

            const send = {
              fileName : image.name,
              fileType : image.type,
              fileData : encodedFile,
            }
            alert("Trying to fetch...")
            const result = await fetch('http://localhost:5000/sendImage', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json' 
              }, 
              body: JSON.stringify(send)
            })
            alert("fetched.")

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

      if (response){
        alert("REsponse")
        alert(JSON.stringify(response))
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
    <div className="App">
      <h1>Welcome to Car Classifier</h1>
      <h3>CS4342 Final Project - Cam Norris, Akaash Walker</h3>
      <h2>Upload a picture of your car <b>from the side </b> and find out what type of car it is</h2>
      <h2>(SUV, Sudan, Coupe, Truck, Semi)</h2>
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
