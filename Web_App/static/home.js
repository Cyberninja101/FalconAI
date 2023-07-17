// $(function() {
//     $("submit_button").on('click', function(e) {
//         e.preventDefault()
//         $.getJSON('/filter/top',
//            {
//         // do nothing
//         });
//     });
// });

var form_entry = document.getElementById("entry");
var chat_box = document.getElementById("chat_log");
var button = document.getElementById("submit_button");
var test_button = document.getElementById("test_but");
var test = document.getElementById("test");
var test2 = document.getElementById("test2");
var count = 0;
// let date_time3 = date_time2.getTime();
//false = user turn, true = machine turn
var turn = Boolean(false);

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
            response = request.responseText;

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

