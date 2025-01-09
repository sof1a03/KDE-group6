import { Component, OnInit } from '@angular/core';
import { Input } from '@angular/core';
import { BookService } from '../../book-service';
import { Book } from '../../book-service';

@Component({
  selector: 'app-book-page',
  imports: [],
  standalone: true,
  templateUrl: './book-page.component.html',
  styleUrl: './book-page.component.css'
})

export class BookPageComponent implements OnInit {
  @Input() id= '';

  book = {} as Book;

  constructor(private bookService: BookService) { }

  ngOnInit(){
    this.bookService.getBook(this.id).subscribe(book => {
      this.book = book;
      console.log(book);
    });
  }
}
