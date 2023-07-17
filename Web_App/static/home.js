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

//false = user turn, true = machine turn
var turn = Boolean(false);

// Require jQuery
const scrollSmoothlyToBottom = (id) => {
    const element = $(`#${id}`);
    element.animate({
       scrollTop: element.prop("scrollHeight")
    }, 500);
 }

// var response = "";
function upload(){
    // uploads user chat to chatlog, POSTs to flask, 
    // gets model response, uploads model response to chatlog
    // boolean flag
    var enter_flag = new Boolean(false);
    const request = new XMLHttpRequest();
    var form_entry = document.getElementById("entry").value;
    response = "";
    request.open('POST', `/new_entry/${form_entry}`);

    // creating user chat
    var div = document.createElement("div");

    // make sure div is not empty, takes care of the spaces
    if (!/\S/.test(form_entry)) {
        // Didn't find something other than a space which means it's empty
        enter_flag = true;
    }
    else if (turn==false){
        // make user chat
        div.id = "user_chat";
        const node = document.createTextNode(form_entry);
        div.appendChild(node);
        chat_box.appendChild(div);
        turn = true; // Set turn to machine
        scrollSmoothlyToBottom("chat_log")
        
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

