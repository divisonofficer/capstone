let chat_list = [
  /*{

    owner : 0,
    type : 0,
    text : "하이루?",
    time : "몰? 루?",
}*/
];

function profile_other() {
  return `
    <div class="profile"> 
        <img src="https://img.hankyung.com/photo/201903/AA.19067065.1.jpg" alt="" onerror="this.style.backgroundColor = 'white';"/>
        <text>라이언</text>
    </div>    
    `;
}

function profile_me() {
  return `
    <div class="profile"> 
        <img src="https://mblogthumb-phinf.pstatic.net/MjAxNzAzMjdfODIg/MDAxNDkwNjEwNDA0MzM0.c4SZEA5JFpJcc40a-l2EqRVpjtg2hk57F0NJER3yXoEg.I4JHmmJgg7hxe-bs0CvJkm9FgClJ3am8y8NjTFa420Ug.JPEG.achika0123/muji.jpg?type=w800" alt="" onerror="this.style.backgroundColor = 'white';""/>
        <text>무지</text>
    </div>    
    `;
}

function msg_me(time, text, timeVisible = true) {
  return `
        <div class="message from-me">
            <div class="time">${timeVisible ? time : ""}</div>
            <div class="text">${text.replaceAll("\n", "<br>")}</div>
        </div>
    `;
}

function msg_other(time, text, timeVisible = true) {
  return `
        <div class="message from-other">
            <div class="time">${timeVisible ? time : ""}</div>
            <div class="text">${text.replaceAll("\n", "<br>")}</div>
        </div>
    `;
}

function chatDateHead() {
  return `
    <div style="display: flex; justify-content: center;">
        <div class="date-container">
            <span>${new Date().toLocaleDateString([], {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}</span>
        </div>
    </div>
    `;
}

function renderChat() {
  $("#chat-left").empty();
  $("#chat-right").empty();
  $("#chat-left").append(chatDateHead());
  $("#chat-right").append(chatDateHead());

  let i = 0,
    len = chat_list.length;
  chat_list.forEach((chat) => {
    const profileVisible = i == 0 || chat_list[i - 1].owner != chat.owner;
    const timeVisible =
      i == len - 1 ||
      chat_list[i + 1].owner != chat.owner ||
      chat_list[i + 1].time != chat.time;
    if (chat.owner == 0) {
      $("#chat-left").append(msg_me(chat.time, chat.text, timeVisible));
      if (profileVisible) {
        $("#chat-right").append(profile_other());
      }
      $("#chat-right").append(msg_other(chat.time, chat.text, timeVisible));
    } else {
      if (profileVisible) {
        $("#chat-left").append(profile_me());
      }
      $("#chat-left").append(msg_other(chat.time, chat.text, timeVisible));
      $("#chat-right").append(msg_me(chat.time, chat.text, timeVisible));
    }

    i = i + 1;
  });
}

function sendMsg(owner) {
  let text = $("#input-left").val();
  if (owner == 1) {
    text = $("#input-right").val();
  }
  text = text.trim();
  if (text == "") {
    return;
  }

  if (owner == 1) {
    $("#input-right").val("");
    $("#send-right").css("background-color", "#e0e0e0");
  } else {
    $("#input-left").val("");
    $("#send-left").css("background-color", "#e0e0e0");
  }

  chat_list.push({
    owner: owner,
    type: 0,
    text: text,
    // time with hour and minute
    time: new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
  });
  renderChat();
  $("#chat-right").scrollTop($("#chat-right")[0].scrollHeight);
  $("#chat-left").scrollTop($("#chat-left")[0].scrollHeight);
}

$(document).ready(() => {
  renderChat();

  //if input is empty, make send button disabled with gray color
  $("#input-left").on("input", function () {
    if ($(this).val().trim() == "") {
      $("#send-left").css("background-color", "#e0e0e0");
    } else {
      $("#send-left").css("background-color", "#ffeb3b");
    }
  });

  $("#input-right").on("input", function () {
    if ($(this).val().trim() == "") {
      $("#send-right").css("background-color", "#e0e0e0");
    } else {
      $("#send-right").css("background-color", "#ffeb3b");
    }
  });

  $("#input-left").keyup(function (e) {
    if (e.which == 13 && !e.shiftKey) {
      sendMsg(0);
      $("#input-left").blur();
      setTimeout(() => {
        $("#input-left").focus();
      }, 10);
    }
  });

  $("#input-right").keyup(function (e) {
    if (e.which == 13 && !e.shiftKey) {
      sendMsg(1);
      $("#input-right").blur();
      setTimeout(() => {
        $("#input-right").focus();
      }, 10);
    }
  });
});
