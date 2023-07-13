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

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('button').forEach(button => {
        button.onclick = () => {
            const request = new XMLHttpRequest();
            request.open('POST', `/new_entry`);
            request.onload = () => {
                var div = document.createElement("div");
                div.id = "user_chat";
                const node = document.createTextNode(form_entry.value);
                div.appendChild(node);
                chat_box.appendChild(div);
                // chat_box.scrollBy(0, div.scrollHeight);
            }; 
            request.send();
        };
    });
    
});

var chat_box = document.getElementById("chat_log");
var user_chat = document.getElementById("user_chat");

function Scroller (){
    
}

