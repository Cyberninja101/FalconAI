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
    const request = new XMLHttpRequest();
    var form_entry = document.getElementById("entry").value;
    request.open('POST', `/new_entry/${form_entry}`);
    request.onload = () => {
        // response is what the flask function returns
        const response = request.responseText;
        var div = document.createElement("div");
        div.id = "user_chat";
        const node = document.createTextNode(form_entry);
        div.appendChild(node);
        chat_box.appendChild(div);
    }; 
    request.send();
    chat_box.scrollTo(0, chat_box.scrollHeight);
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

