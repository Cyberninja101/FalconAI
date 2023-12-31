var form_entry = document.getElementById("entry");
var chat_box = document.getElementById("chat_log");
var count = 0;
var checkbox = document.querySelector("input[name=color_mode]");
var turn = Boolean(false);
var mode = "normal";
var hmt_bool_flag = 1;

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

function onload_do(){
    document.getElementById("pdf_upload").style.display = "none";
    document.getElementById("uploaded_files").style.display = "none";
}

// Hamburger menu
function menuOnClick() {
    document.getElementById("menu-bar").classList.toggle("change");
    document.getElementById("nav").classList.toggle("change");
    document.getElementById("menu-bg").classList.toggle("change-bg");
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
    request.open('POST', `/new_entry/${mode}/${stringToHex(form_entry)}`);

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
document.getElementById("submit_button").addEventListener("click", upload)

//check if enter button is hit 
function enter_check (e){
    var key_pressed = e.key;
    if (key_pressed == "Enter"){
        upload();
    }
}

function reset() {
    $('input[type="checkbox"]').each(function(){
        $(this).prop('checked', false);
    });
    location.reload()
}

checkbox.addEventListener("change", function() {
    if (this.checked) {
        // Document mode
        document.getElementById("pdf_upload").style.display = "flex";
        document.getElementById("uploaded_files").style.display = "initial";

        var mode_text = document.createElement("div");
        mode_text.id = "mode_text";
        var mode_node = document.createTextNode("Document Mode");
        mode_text.appendChild(mode_node);
        chat_box.appendChild(mode_text);
        scrollSmoothlyToBottom("chat_log")

        mode = "document";
    } else {
        // Normal Mode
        document.getElementById("pdf_upload").style.display = "none";
        document.getElementById("uploaded_files").style.display = "none";

        var mode_text = document.createElement("div");
        mode_text.id = "mode_text";
        var mode_node = document.createTextNode("Normal Mode");
        mode_text.appendChild(mode_node);
        chat_box.appendChild(mode_text);
        scrollSmoothlyToBottom("chat_log")

        mode = "normal";
    }
});

function hmt(){ 
    if (hmt_bool_flag == 1){
        console.log("hmt pressed")
        document.getElementById("HMT_img").src="../static/images/HMT_robot_icon_blue.png";
        hmt_bool_flag = 0;
        // document.querySelector('input[name=color_mode]').style.display = "none";
        mode = "hmt";
    } else if (hmt_bool_flag==0){
        console.log("elsed")
        document.getElementById("HMT_img").src="../static/images/HMT_robot_icon.png";
        hmt_bool_flag=1;
        // document.querySelector('btn-color-mode-switch').style.display = "inline-block";
        mode = "normal";
    }
}

document.getElementById("HMT_btn").addEventListener("click", hmt)

document.getElementById("pdf_upload").addEventListener('change', function(e) {
    if (e.target.files[0]) {
        var file = document.createElement("div");
        file.id = "pdf_files";
        var txt_node_form = document.createTextNode(String(e.target.files[0].name))
        file.appendChild(txt_node_form);
        document.getElementById("uploaded_files").appendChild(file);
    }
});
