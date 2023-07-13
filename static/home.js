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

function upload(){
    // boolean flag
    var enter_flag = new Boolean(false);
    const request = new XMLHttpRequest();
    var form_entry = document.getElementById("entry").value;
    request.open('POST', `/new_entry/${form_entry}`);
    request.onload = () => {
        // response is what the flask function returns
        const response = request.responseText;
        var div = document.createElement("div");

        // make sure div is not empty, takes care of the spaces
        if (!/\S/.test(form_entry)) {
            // Didn't find something other than a space which means it's empty
            enter_flag = true;
        }
        else {
            div.id = "user_chat";
            const node = document.createTextNode(form_entry);
            div.appendChild(node);
            chat_box.appendChild(div);
        }

        
    }; 
    // make sure div is not empty
    
    if (enter_flag == false) {
        request.send();
        // change scroll, it doesn't work
        chat_box.scrollTop = chat_box.scrollTopMax;
        document.getElementById("entry").value = "";
    }
        
    
    
    
}

//check if button is manually hit, check this if we add another button
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('button').forEach(button => {
        button.onclick = () => {
            upload();
        };
    });
});

//check if enter button is hit 
function enter_check (e){
    var key_pressed = e.key;
    if (key_pressed == "Enter"){
        upload();
    }
}

