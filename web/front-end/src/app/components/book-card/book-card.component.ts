import { Component, Input } from '@angular/core';
import { UserlikeService } from '../../userlike.service';
import { UserService } from '../../user.service';
import { OnInit } from '@angular/core';

@Component({
  selector: 'app-book-card',
  standalone: true,
  imports: [],
  templateUrl: './book-card.component.html',
  styleUrl: './book-card.component.css'
})

export class BookCardComponent implements OnInit{
  constructor(
    private userLikeService: UserlikeService,
  private userService: UserService) {}
  @Input() image_url = '';
  @Input() title= '';
  @Input() publisher= '';
  @Input() year= 0;
  @Input() ISBN= '';
  @Input() bookid= '';
  @Input() genres= [''];
  @Input() author= '';

  getAfterHash(str: string) {
    const hashIndex = str.indexOf('#');
    if (hashIndex === -1) {
      return str;
    }
    return str.substring(hashIndex + 1);
  }


  ngOnInit(){
    this.bookid = this.getAfterHash(this.bookid)
  }

  onLike() {
    const username = this.userService.username;
    if (username !== null){
      this.userLikeService.like(username, this.bookid, this.title);
    }
  }

}
