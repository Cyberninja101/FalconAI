// $(function() {
//     $("submit_button").on('click', function(e) {
//         e.preventDefault()
//         $.getJSON('/filter/top',
//            {
//         // do nothing
//         });
//     });
// });

// Hamburger menu
function menuOnClick() {
    document.getElementById("menu-bar").classList.toggle("change");
    document.getElementById("nav").classList.toggle("change");
    document.getElementById("menu-bg").classList.toggle("change-bg");
}

var form_entry = document.getElementById("entry");
var chat_box = document.getElementById("chat_log");
var button = document.getElementById("submit_button");
var count = 0;
var checkbox = document.querySelector("input[name=color_mode]");
// let date_time3 = date_time2.getTime();
//false = user turn, true = machine turn
var turn = Boolean(false);

function onload_do(){
    document.getElementById("pdf_upload").style.display = "none";
    document.getElementById("uploaded_files").style.display = "none";
}

checkbox.addEventListener("change", function() {
    if (this.checked) {
        document.getElementById("pdf_upload").style.display = "flex";
    } else {
        document.getElementById("pdf_upload").style.display = "none";
        document.getElementById("uploaded_files").style.display = "none";
    }
});

// Require jQuery
const scrollSmoothlyToBottom = (id) => {
    const element = $(`#${id}`);
    element.animate({
       scrollTop: element.prop("scrollHeight")
    }, 500);
 }
 
const stringToHex = (str) => {
    let hex = '';
    for (let i = 0; i < str.length; i++) {
        const charCode = str.charCodeAt(i);
        var hexValue = charCode.toString(16);
        var hexValue = "," + hexValue
        // Pad with zeros to ensure two-digit representation
        hex += hexValue;
    }
    return hex;
}

const hexToString = (hex) => {
    let str = '';
    for (let i = 0; i < hex.length; i += 2) {
        const hexValue = hex.substr(i, 2);
        const decimalValue = parseInt(hexValue, 16);
        str += String.fromCharCode(decimalValue);
    }
    return str;
};

document.getElementById("pdf_upload").addEventListener('change', function(e) {
    if (e.target.files[0]) {
        // var fs = require('fs');
        // var files = fs.readdirSync("contexts/");
        // console.log(files)
        var file = document.createElement("div");
        file.id = "pdf_files";
        var txt_node_form = document.createTextNode(String(e.target.files[0].name))
        file.appendChild(txt_node_form);
        document.getElementById("uploaded_files").appendChild(file);
    }
    document.getElementById("uploaded_files").style.display = "initial";
  });

async function reset_button(){
    const fs = require("fs");
    const path = require("path");

    console.log("works")

    const directory = "Web_App/context/";

    // for (const file of await fs.readdir(directory)) {
    //     await fs.unlink(path.join(directory, file));
    // }

    window.location.reload();
}

function upload(){
    // uploads user chat to chatlog, POSTs to flask, 
    // gets model response, uploads model response to chatlog
    // boolean flag
    var enter_flag = new Boolean(false);
    const request = new XMLHttpRequest();
    var form_entry = String(document.getElementById("entry").value);
    console.log(form_entry)
    response = "";
    request.open('POST', `/new_entry/${stringToHex(form_entry)}`);

    // creating user chat
    var div = document.createElement("div");

    // make sure div is not empty, takes care of the spaces
    if (!/\S/.test(form_entry)) {
        // Didn't find something other than a space which means it's empty
        enter_flag = true;
    }
    else {
        if (turn==false){
            var current_time = new Date().getTime();
            if ((current_time - count) >= 1000){
                div.id = "user_chat";
                var node = document.createTextNode(form_entry);
                div.appendChild(node);
                chat_box.appendChild(div);
                turn = true; // Set turn to machine
                scrollSmoothlyToBottom("chat_log")
                count = new Date().getTime();
            }
        }   
    }

    request.onload = () => {
        // response is what the flask function returns
        if (turn) {
            if (/best barbeque/i.test(form_entry) || /best b.b.q/i.test(form_entry) || /best bbq/i.test(form_entry)){
                response = "Kloby's is the best barbeque";
            } else{
                response = request.responseText;
            }
            

            // falconAI response
            var div = document.createElement("div");
            div.id = "ai_chat";
            const node = document.createTextNode(response);
            div.appendChild(node);
            chat_box.appendChild(div);
            
            // document.getElementById("entry").value = "";
    
            turn = false; // Set turn to human
            scrollSmoothlyToBottom("chat_log")
        }
    }; 

    // make sure div is not empty
    
    if (enter_flag == false) {
        request.send();
        
        
        document.getElementById("entry").value = "";
    }    
    
}

//check if button is manually hit, check this if we add another button
button.addEventListener("click", upload)


//check if enter button is hit 
function enter_check (e){
    var key_pressed = e.key;
    if (key_pressed == "Enter"){
        upload();
    }
}

