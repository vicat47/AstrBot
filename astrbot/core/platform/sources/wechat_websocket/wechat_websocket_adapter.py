import asyncio
import json
import os
import traceback
from asyncio import Queue
from datetime import datetime
import time
from enum import Enum
from typing import Awaitable, Any

import aiohttp
import websockets

from astrbot import logger
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.platform import Platform, AstrMessageEvent, PlatformMetadata, AstrBotMessage, MessageType, \
    MessageMember
from astrbot.core.platform.message_session import MessageSesion
from astrbot.core.platform.sources.wechat_websocket.wechat_websocket_message_event import WeChatWebsocketMessageEvent


class WechatWebsocketMessageType(Enum):
    HEART_BEAT = 5005
    RECV_TXT_MSG = 1
    RECV_PIC_MSG = 3
    USER_LIST = 5000
    GET_USER_LIST_SUCCSESS = 5001
    GET_USER_LIST_FAIL = 5002
    TXT_MSG = 555
    PIC_MSG = 500
    AT_MSG = 550
    CHATROOM_MEMBER = 5010
    CHATROOM_MEMBER_NICK = 5020
    PERSONAL_INFO = 6500
    DEBUG_SWITCH = 6000
    PERSONAL_DETAIL = 6550
    DESTROY_ALL = 9999
    # 微信好友请求消息
    NEW_FRIEND_REQUEST = 37
    # 同意微信好友请求消息
    AGREE_TO_FRIEND_REQUEST = 10000
    ATTATCH_FILE = 5003
    # 啥都有，包括公众号
    CHAOS_TYPE = 49

# TODO: 完善代码
class WeChatWebsocketAdapter(Platform):

    def __init__(
        self,
        platform_config: dict,
        platform_settings: dict,
        event_queue: Queue
    ) -> None:
        super().__init__(event_queue)
        self.config = platform_config
        self.settings = platform_settings

        self.unique_session = platform_settings["unique_session"]

        self.metadata = PlatformMetadata(
            name="wechat-websocket",
            description="微信机器人个人适配",
            id=self.config.get("id", "wechat-websocket"),
            support_streaming_message=False,
        )

        self.host = self.config.get("host")
        self.port = self.config.get("port")
        self.active_message_poll: bool = self.config.get(
            "ww_active_message_poll",
            False,
        )
        self.active_message_poll_interval: int = self.config.get(
            "ww_active_message_poll_interval",
            5,
        )
        self.base_url = f"http://{self.host}:{self.port}"
        self.wxid = None
        self.nickname = None
        self.head_pic = None
        self.ws_handle_task = None


        self.client_id = platform_config["client_id"]
        self.client_secret = platform_config["client_secret"]

    async def run(self) -> Awaitable[Any]:
        """启动平台适配器的运行实例。"""
        logger.info("WeChatWebsocket 适配器正在启动...")

        isLoginIn = await self.check_online_status()
        if (isLoginIn):
            self.ws_handle_task = asyncio.create_task(self.connect_websocket())
        pass

    async def terminate(self):
        return await super().terminate()

    def meta(self) -> PlatformMetadata:
        return PlatformMetadata(
            name="wechat_websocket",
            description="微信机器人个人适配",
            id=self.config.get("id")
        )

    async def send_by_session(self,
                              session: MessageSesion,
                              message_chain: MessageChain):
        dummy_message_obj = AstrBotMessage()
        dummy_message_obj.session_id = session.session_id
        # 根据 session_id 判断消息类型
        if "@chatroom" in session.session_id:
            dummy_message_obj.type = MessageType.GROUP_MESSAGE
            if "#" in session.session_id:
                dummy_message_obj.group_id = session.session_id.split("#")[0]
            else:
                dummy_message_obj.group_id = session.session_id
            dummy_message_obj.sender = MessageMember(user_id="", nickname="")
        else:
            dummy_message_obj.type = MessageType.FRIEND_MESSAGE
            dummy_message_obj.group_id = ""
            dummy_message_obj.sender = MessageMember(user_id="", nickname="")
        sending_event = WeChatWebsocketMessageEvent(
            message_str="",
            message_obj=dummy_message_obj,
            platform_meta=self.meta(),
            session_id=session.session_id,
            adapter=self,
        )
        await sending_event.send(message_chain)

    def commit_event(self, event: AstrMessageEvent):
        super().commit_event(event)

    def get_client(self):
        super().get_client()

    async def check_online_status(self):
        url = f"{self.base_url}/api/get_personal_info"
        params = {
            "para": {
                "id": f"{int(datetime.now().timestamp()*1000)}",
                "type": 6500,
                "roomid": "",
                "wxid": "",
                "content": "",
                "nickname": "",
                "ext": ""
            }
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    response_data = await response.json()
                    if response.status == 200 and response_data.get("status") == "SUCCSESSED":
                        person_info = json.loads(response_data.get("content"))
                        self.wxid = person_info.get("wx_id")
                        self.nickname = person_info.get("wx_name")
                        self.head_pic = person_info.get("wx_head_image")
                        return True
            except aiohttp.ClientConnectorError as e:
                logger.error(f"连接到 wechat_websocket 服务失败: {e}")
                return False
            except Exception as e:
                logger.error(f"检查 wechat_websocket 在线状态时发生错误: {e}")
                logger.error(traceback.format_exc())
                return False

    async def connect_websocket(self):
        os.environ["no_proxy"] = f"localhost,127.0.0.1,{self.host}"
        ws_url = f"ws://{self.host}:{self.port}"
        logger.info(f"{self.metadata.name} 正在连接 WebSocket: {ws_url}")
        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    logger.debug(f"{self.metadata.name} WebSocket 连接成功。")
                    wait_time = (
                        self.active_message_poll_interval
                        if self.active_message_poll
                        else 120
                    )
                    while True:
                        try:
                            message = await asyncio.wait_for(ws.recv(), wait_time)
                            asyncio.create_task(self.handle_websocket_message(message))
                        except asyncio.TimeoutError:
                            logger.debug(f"WebSocket 连接空闲超过 {wait_time} s")
                            break
                        except websockets.exceptions.ConnectionClosedOK:
                            logger.info("WebSocket 连接正常关闭。")
                            break
                        except Exception as e:
                            logger.error(f"处理 WebSocket 消息时发生错误: {e}")
                            break
            except Exception as e:
                logger.error(
                    f"WebSocket 连接失败: {e}, 请检查{self.metadata.name}服务状态，或尝试重启{self.metadata.name}适配器。",
                )
                await asyncio.sleep(5)

    async def handle_websocket_message(self, message: str):
        """处理从 WebSocket 接收到的消息。"""
        logger.debug(f"收到 WebSocket 消息: {message}")
        try:
            message_data = json.loads(message)
            if (message_data.get("id") is not None
                and not (message_data.get("wxid") is None and message_data.get("roomid") is None)
            ):
                abm = await self.convert_message(message_data)
                if abm:
                    message_event = WeChatWebsocketMessageEvent(
                        message_str=abm.message_str,
                        message_obj=abm,
                        platform_meta=self.meta(),
                        session_id=abm.session_id,
                        # 传递适配器实例，以便在事件中调用 send 方法
                        adapter=self,
                    )
                    # 提交事件到事件队列
                    self.commit_event(message_event)
            else:
                logger.warning(f"收到未知结构的 WebSocket 消息: {message_data}")
        except json.JSONDecodeError:
            logger.error(f"无法解析 WebSocket 消息为 JSON: {message}")
        except Exception as e:
            logger.error(f"处理 WebSocket 消息时发生错误: {e}")
        pass

    async def convert_message(self, raw_message: dict) -> AstrBotMessage | None:
        """将 WeChatPadPro 原始消息转换为 AstrBotMessage。"""
        abm = AstrBotMessage()
        abm.raw_message = raw_message
        abm.message_id = str(raw_message.get("id"))
        abm.timestamp = int(time.mktime(time.strptime(raw_message.get("time"), "%Y-%m-%d %H:%M:%S")))
        abm.self_id = self.wxid

        if int(time.time()) - abm.timestamp > 180:
            logger.warning(
                f"忽略 3 分钟前的旧消息：消息时间戳 {abm.timestamp} 超过当前时间 {int(time.time())}。",
            )
            return None


        from_user_id = raw_message.get("wxid", {}).get("str", "")
        content = raw_message.get("content", {}).get("str", "")
        msg_type = raw_message.get("type")

        abm.message_str = ""
        abm.message = []

        if from_user_id == self.wxid:
            logger.info("忽略来自自己的消息。")
            return None

        if from_user_id in ["weixin", "newsapp", "newsapp_wechat"]:
            logger.info("忽略来自微信团队的消息。")
            return None
        if await self._process_chat_type(
            abm,
            raw_message,
            from_user_id,
            content,
        ):
            await self._process_message_content(abm, raw_message, msg_type, content)
            return abm
        return None

    async def _process_chat_type(self, abm, raw_message: dict, from_user_id, content):
        """判断消息是群聊还是私聊，并设置 AstrBotMessage 的基本属性。"""
        if from_user_id == "weixin":
            return False
        at_me = False
        if "@chatroom" in from_user_id:
            abm.type = MessageType.GROUP_MESSAGE
            abm.group_id = from_user_id

            sender_wxid = raw_message.get("id1")
            abm.sender = MessageMember(user_id=sender_wxid, nickname="")

            # 获取群聊发送者的nickname
            if sender_wxid:
                accurate_nickname = await self._get_group_member_nickname(
                    abm.group_id,
                    sender_wxid,
                )
                if accurate_nickname:
                    abm.sender.nickname = accurate_nickname

            # 对于群聊，session_id 可以是群聊 ID 或发送者 ID + 群聊 ID (如果 unique_session 为 True)
            if self.unique_session:
                abm.session_id = f"{from_user_id}#{abm.sender.user_id}"
            else:
                abm.session_id = from_user_id

            # todo 处理 xml 内容
            raw_message.get("other")
        pass

